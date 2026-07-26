"""Top-level workflow navigation components."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageViews:
    """Named containers used by the workflow-specific panels."""

    tabs: object
    preparation_tabs: object
    preparation: object
    setup: object
    analysis: object
    export: object
    downsample: object
    gabor: object
    wavelet: object


def create_stage_views(ctk, tk, parent, *, palette, outer_labels, preparation_labels) -> StageViews:
    """Build the nested workflow navigation and return named tab containers."""
    tabs = ctk.CTkTabview(
        parent,
        height=740,
        corner_radius=8,
        fg_color=palette["frame_color"],
        segmented_button_fg_color=palette["choice_bg"],
        segmented_button_selected_color=palette["choice_btn"],
        segmented_button_selected_hover_color=palette["choice_hover"],
        segmented_button_unselected_color="#E2E8F0",
        segmented_button_unselected_hover_color="#CBD5E1",
        text_color=palette["text_color"],
    )
    tabs.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
    preparation = tabs.add(outer_labels[0])
    setup = tabs.add(outer_labels[1])
    analysis = tabs.add(outer_labels[2])
    export = tabs.add(outer_labels[3])

    preparation_tabs = ctk.CTkTabview(
        preparation,
        corner_radius=8,
        fg_color="#F8FAFC",
        segmented_button_fg_color=palette["choice_bg"],
        segmented_button_selected_color=palette["choice_btn"],
        segmented_button_selected_hover_color=palette["choice_hover"],
        segmented_button_unselected_color="#E2E8F0",
        segmented_button_unselected_hover_color="#CBD5E1",
        text_color=palette["text_color"],
    )
    preparation_tabs.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
    downsample = preparation_tabs.add(preparation_labels[0])
    gabor = preparation_tabs.add(preparation_labels[1])
    wavelet = preparation_tabs.add(preparation_labels[2])
    return StageViews(tabs, preparation_tabs, preparation, setup, analysis, export, downsample, gabor, wavelet)
