"""Reusable configuration-field widgets."""

from __future__ import annotations


def create_text_entry_row(
    ctk,
    parent,
    *,
    label: str,
    value: object,
    hint: str,
    row: int,
    text_color: str,
    muted_text: str,
    frame_color: str,
):
    """Create a labelled text input row and return its label, wrapper, and entry."""
    label_widget = ctk.CTkLabel(
        parent,
        text=label,
        text_color=text_color,
        font=ctk.CTkFont(size=12),
    )
    label_widget.grid(row=row, column=0, sticky="w", pady=4)

    entry_wrap = ctk.CTkFrame(parent, fg_color="transparent")
    entry_wrap.grid(row=row, column=1, pady=3, padx=(10, 0), sticky="ew")
    entry_wrap.columnconfigure(0, weight=1)

    entry = ctk.CTkEntry(
        entry_wrap,
        height=30,
        corner_radius=6,
        border_width=1,
        fg_color=frame_color,
        text_color=text_color,
        placeholder_text_color=muted_text,
        border_color="#CBD5E1",
        placeholder_text=hint,
    )
    entry.insert(0, str(value))
    entry.grid(row=0, column=0, sticky="ew")
    return label_widget, entry_wrap, entry
