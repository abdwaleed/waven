"""Add missing Google-style docstrings to Python source files.

This script is intentionally conservative: it only inserts docstrings where a
module, class, function, or async function has no docstring at all. Existing
docstrings are left untouched so scientific descriptions written by humans do
not get flattened into generated text.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path


EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", "site", "build", "dist"}


@dataclass
class Insertion:
    """Text insertion requested for a Python source file.

    Args:
        line: One-based source line before which the docstring should be added.
        text: Docstring text to insert, including trailing newlines.
    """

    line: int
    text: str


def _is_public_name(name: str) -> bool:
    """Return whether a Python object name is intended as public API.

    Args:
        name: Function or class name from the syntax tree.

    Returns:
        True when the name is public or a dunder method; otherwise False.
    """

    return not name.startswith("_") or (name.startswith("__") and name.endswith("__"))


def _summary_for(kind: str, name: str) -> str:
    """Build a concise summary line for a generated docstring.

    Args:
        kind: Human-readable object kind.
        name: Object name.

    Returns:
        A short sentence suitable as the first docstring line.
    """

    clean = name.strip("_").replace("_", " ") or name
    return f"{kind} for {clean}."


def _module_summary(path: Path) -> str:
    """Build a module summary from a filesystem path.

    Args:
        path: Python file being documented.

    Returns:
        One-sentence module summary.
    """

    stem = path.stem.replace("_", " ")
    if path.name == "__init__.py":
        stem = path.parent.name.replace("_", " ")
    return f"{stem.title()} module."


def _argument_names(node: ast.AST) -> list[str]:
    """Collect documented argument names for a function node.

    Args:
        node: Function or async-function syntax tree node.

    Returns:
        Argument names excluding ``self`` and ``cls``.
    """

    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    args = []
    arguments = node.args
    ordered = [
        *arguments.posonlyargs,
        *arguments.args,
        *arguments.kwonlyargs,
    ]
    if arguments.vararg is not None:
        ordered.append(arguments.vararg)
    if arguments.kwarg is not None:
        ordered.append(arguments.kwarg)
    for arg in ordered:
        if arg.arg not in {"self", "cls"}:
            args.append(arg.arg)
    return args


def _has_return_value(node: ast.AST) -> bool:
    """Return whether a function contains an explicit value return.

    Args:
        node: Function or async-function syntax tree node.

    Returns:
        True when the body includes ``return <value>``.
    """

    return any(isinstance(child, ast.Return) and child.value is not None for child in ast.walk(node))


def _indent_for_line(lines: list[str], line_no: int) -> str:
    """Return the indentation used at a specific source line.

    Args:
        lines: Source file split into lines.
        line_no: One-based line number.

    Returns:
        Leading whitespace for the requested line.
    """

    line = lines[line_no - 1]
    return line[: len(line) - len(line.lstrip())]


def _docstring_block(lines: list[str], node: ast.AST, path: Path) -> str:
    """Create a generated Google-style docstring for a syntax node.

    Args:
        lines: Source file split into lines.
        node: Module, class, function, or async-function node.
        path: Python source file being edited.

    Returns:
        Fully formatted docstring block with indentation.
    """

    if isinstance(node, ast.Module):
        indent = ""
        content = [_module_summary(path)]
    elif isinstance(node, ast.ClassDef):
        insert_line = node.body[0].lineno
        indent = _indent_for_line(lines, insert_line)
        content = [_summary_for("Container", node.name)]
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        insert_line = node.body[0].lineno
        indent = _indent_for_line(lines, insert_line)
        content = [_summary_for("Function", node.name)]
        args = _argument_names(node)
        if args:
            content.extend(["", "Args:"])
            content.extend([f"    {arg}: Input value for this operation." for arg in args])
        if _has_return_value(node) and node.name != "__init__":
            content.extend(["", "Returns:", "    Result produced by the operation."])
    else:
        raise TypeError(f"Unsupported node type: {type(node)!r}")

    if len(content) == 1:
        return f'{indent}"""{content[0]}"""\n'
    body = [f'{indent}"""' + content[0]]
    body.extend(f"{indent}{line}" if line else "" for line in content[1:])
    body.append(f'{indent}"""')
    return "\n".join(body) + "\n"


def _module_insert_line(lines: list[str]) -> int:
    """Find where to insert a module docstring.

    Args:
        lines: Source file split into lines.

    Returns:
        One-based insertion line.
    """

    line = 1
    if lines and lines[0].startswith("#!"):
        line = 2
    if len(lines) >= line and "coding" in lines[line - 1]:
        line += 1
    return line


def collect_insertions(path: Path) -> list[Insertion]:
    """Collect missing docstrings for a source file.

    Args:
        path: Python source file to inspect.

    Returns:
        Insertions needed to add missing docstrings.
    """

    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    insertions: list[Insertion] = []
    if ast.get_docstring(tree) is None:
        insertions.append(
            Insertion(
                _module_insert_line(lines),
                _docstring_block(lines, tree, path),
            )
        )
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if ast.get_docstring(node) is not None:
                continue
            if not node.body:
                continue
            insertions.append(
                Insertion(
                    node.body[0].lineno,
                    _docstring_block(lines, node, path),
                )
            )
    return sorted(insertions, key=lambda item: item.line, reverse=True)


def apply_insertions(path: Path, insertions: list[Insertion]) -> None:
    """Apply docstring insertions to a Python source file.

    Args:
        path: Python source file to edit.
        insertions: Insertion records sorted or unsorted.
    """

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for insertion in sorted(insertions, key=lambda item: item.line, reverse=True):
        lines.insert(insertion.line - 1, insertion.text)
    path.write_text("".join(lines), encoding="utf-8")


def iter_python_files(root: Path, include_legacy: bool) -> list[Path]:
    """List Python files that should receive generated docstrings.

    Args:
        root: Repository root.
        include_legacy: Whether to include root-level ``old_*.py`` files.

    Returns:
        Sorted Python file paths.
    """

    files = []
    for path in root.rglob("*.py"):
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        if not include_legacy and path.name.startswith("old_"):
            continue
        files.append(path)
    return sorted(files)


def main() -> int:
    """Run the docstring insertion command-line interface.

    Returns:
        Process exit code.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root to scan.")
    parser.add_argument("--check", action="store_true", help="Report files that need docstrings without editing.")
    parser.add_argument("--include-legacy", action="store_true", help="Include root-level old_*.py snapshots.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    needed: list[tuple[Path, int]] = []
    for path in iter_python_files(root, include_legacy=args.include_legacy):
        insertions = collect_insertions(path)
        if not insertions:
            continue
        needed.append((path, len(insertions)))
        if not args.check:
            apply_insertions(path, insertions)

    for path, count in needed:
        print(f"{path.relative_to(root)}: {count} missing docstring(s)")
    return 1 if args.check and needed else 0


if __name__ == "__main__":
    raise SystemExit(main())
