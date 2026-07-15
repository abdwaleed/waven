"""Tests for the conservative synchronous multi-GPU selection policy."""

from types import SimpleNamespace

from waven.runtime import performance
from waven.runtime.performance import select_compatible_multi_gpu_ids


def _gpu(device_id, name, capability, score, vram_gib):
    return {
        "id": device_id,
        "name": name,
        "capability": capability,
        "throughput_score": score,
        "vram_bytes": vram_gib * 1024**3,
    }


def test_mismatched_architectures_are_not_combined_in_dataparallel():
    ids, rejected = select_compatible_multi_gpu_ids(
        (
            _gpu(0, "RTX 3060 Ti", (8, 6), 38_000, 8),
            _gpu(1, "GTX 1660 Ti", (7, 5), 24_000, 6),
        )
    )

    assert ids == (0,)
    assert rejected and "compute capability" in rejected[0]


def test_similarly_matched_cards_are_retained_for_dataparallel():
    ids, rejected = select_compatible_multi_gpu_ids(
        (
            _gpu(0, "RTX A5000", (8, 6), 64_000, 24),
            _gpu(1, "RTX A5000", (8, 6), 62_000, 24),
        )
    )

    assert ids == (0, 1)
    assert rejected == ()


def test_large_throughput_gap_is_not_combined_even_with_matching_architecture():
    ids, rejected = select_compatible_multi_gpu_ids(
        (
            _gpu(0, "Fast", (8, 6), 80_000, 16),
            _gpu(1, "Slow", (8, 6), 40_000, 16),
        )
    )

    assert ids == (0,)
    assert "throughput" in rejected[0]


def test_gpu_descriptor_accepts_property_objects_without_clock_rate(monkeypatch):
    properties = SimpleNamespace(
        name="Clockless test GPU",
        major=8,
        minor=6,
        total_memory=8 * 1024**3,
        multi_processor_count=20,
    )
    monkeypatch.setattr(
        performance.torch.cuda,
        "get_device_properties",
        lambda _device_id: properties,
    )

    descriptor = performance._gpu_descriptor(0)

    assert descriptor["throughput_score"] == 20
    assert descriptor["capability"] == (8, 6)
