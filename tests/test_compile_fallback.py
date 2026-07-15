"""Tests for safe eager fallback when a lazily compiled runner fails."""

import torch

from waven.wavelets.decomposition import _CompileFallbackRunner


class _FailingCompiledRunner(torch.nn.Module):
    def forward(self, values):
        raise RuntimeError("simulated compiler failure")


def test_compiled_runner_failure_retries_eager_and_stays_eager():
    eager = torch.nn.Identity()
    runner = _CompileFallbackRunner(eager, _FailingCompiledRunner())
    values = torch.tensor([[2.0]])

    assert torch.equal(runner(values), values)
    assert runner._compiled_active is False
    assert torch.equal(runner(values), values)
