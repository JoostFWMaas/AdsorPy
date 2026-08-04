# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Test the SurfaceGeneration class of the `gui.py` module."""

import io
from typing import ParamSpec
from unittest.mock import MagicMock, patch

import pytest
from pydantic import TypeAdapter
from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QLineEdit, QMessageBox
from pytestqt.qtbot import QtBot

from adsorpy.gui import AppState, SurfaceGeneration, SurfaceParameters

P = ParamSpec("P")


@pytest.fixture
def surface_tab(qtbot: QtBot) -> SurfaceGeneration:
    """Instantiate the SurfaceGeneration tab using a real AppState context.

    :param qtbot: Qt instance to mock interaction.
    """
    state = AppState()

    # Pre-populate AppState requirements needed by generate_surface validation checks
    state.seed_input = QLineEdit()
    state.seed_input.setText("42")

    tab = SurfaceGeneration(state)
    qtbot.addWidget(tab)
    return tab


def test_initial_structural_layout_states(surface_tab: SurfaceGeneration) -> None:
    """Verify that layout panels, dropdown configurations, and splitters load correctly.

    :param surface_tab: SurfaceGeneration widget.
    """
    assert surface_tab.main_splitter.count() == 2  # noqa: PLR2004
    assert surface_tab.main_splitter.orientation() == Qt.Orientation.Horizontal

    # Check default combo box configuration entries
    expected_items = sorted(["hexagonal", "square", "honeycomb"])
    actual_items = [surface_tab.surface_dropdown.itemText(i) for i in range(surface_tab.surface_dropdown.count())]
    assert actual_items == expected_items


def test_input_validators_range_boundaries(surface_tab: SurfaceGeneration) -> None:
    """Verify numeric data validation parameters enforce correct logical limits.

    :param surface_tab: SurfaceGeneration widget.
    """
    # Ensure bottom boundaries block zero or negative integer entries
    assert surface_tab._gt_one_validator.bottom() == 1  # noqa: SLF001
    assert surface_tab._pos_float_validator.bottom() == 0.0  # noqa: SLF001


@pytest.mark.parametrize(
    ("geometry_preset", "typed_input", "expected_label_output"),
    [
        ("square", "100", "10000"),  # Straight 1:1 mapping
        ("square", "", "2500"),  # Blank fallback state logic default
        ("hexagonal", "40", "3200"),  # Doubles site count requirement (2x)
        ("honeycomb", "25", "2500"),  # Quadruples site count requirement (4x)
    ],
)
def test_signal_loops_recalculate_real_site_count(
    surface_tab: SurfaceGeneration,
    qtbot: QtBot,
    geometry_preset: str,
    typed_input: str,
    expected_label_output: str,
) -> None:
    """Verify that user interface modifications alter real-time node tracking text.

    :param surface_tab: SurfaceGeneration widget.
    :param qtbot: Mocks interaction.
    :param geometry_preset: the surface type.
    :param typed_input: the input surface count typed by the 'user'.
    :param expected_label_output: the expected output surface count.
    """
    # Alter the dropdown choice
    surface_tab.surface_dropdown.setCurrentText(geometry_preset)

    # Simulate direct physical text keys adjustments
    surface_tab.site_count_input.clear()
    if typed_input:
        qtbot.keyClicks(surface_tab.site_count_input, typed_input)

    # Check that calculations fired instantly via internal Qt signal connections
    assert surface_tab.real_site_count.text() == expected_label_output


def test_generate_surface_success(surface_tab: SurfaceGeneration, qtbot: QtBot) -> None:
    """Verify successful execution updates shared application memory maps and calls the SVG load handle.

    :param surface_tab: SurfaceGeneration widget.
    :param qtbot: Mocks interaction.
    """
    surface_tab.state.seed_input.setText("119")
    surface_tab.lattice_input.setValue(2.45)
    surface_tab.surface_dropdown.setCurrentText("hexagonal")
    site_count = 35
    surface_tab.site_count_input.setText(f"{site_count}")  # Sets base count variable to 35

    # Force calculation to cache baseline count variables into self.surface_count
    surface_tab._get_real_surface_site_count()  # noqa: SLF001

    # Mock out the plotting pipeline and core graphics widget loaders
    with (
        patch("adsorpy.gui.show_surface") as mock_show_surface,
        patch.object(surface_tab.svg_widget, "load") as mock_svg_load,
    ):
        # Simulate physical button interaction
        qtbot.mouseClick(surface_tab.generate_surface_button, Qt.MouseButton.LeftButton)

        # Verify call parameters match application specifications
        mock_show_surface.assert_called_once()
        _, kwargs = mock_show_surface.call_args

        assert kwargs["lattice_a"] == 2.45  # noqa: PLR2004
        assert kwargs["lattice_type"] == "hexagonal"
        assert kwargs["seed"] == 119  # noqa: PLR2004
        assert kwargs["site_count"] == site_count
        assert isinstance(kwargs["filepath"], io.BytesIO)
        assert kwargs["svg_flag"] is True

        # Verify the SVG loader method triggered
        mock_svg_load.assert_called_once()

        # Verify updated models successfully synchronised down into global memory
        adapter = TypeAdapter(SurfaceParameters)
        adapter.validate_python(surface_tab.state.surface_params)
        assert isinstance(surface_tab.state.surface_params, dict)
        assert surface_tab.state.surface_params["lattice_type"] == "hexagonal"  # pyright: ignore[reportTypedDictNotRequiredAccess]
        assert surface_tab.state.surface_params["site_count"] == site_count  # pyright: ignore[reportTypedDictNotRequiredAccess]


def test_generate_surface_handles_malformed_seed_gracefully(surface_tab: SurfaceGeneration, qtbot: QtBot) -> None:
    """Verify that bad seed strings are intercepted before plotting calls execute.

    :param surface_tab: SufaceGeneration widget.
    :param qtbot: Mocks interaction.
    """
    surface_tab.state.seed_input.setText("not_a_number_error")

    # Patch native alert mechanisms to prevent blocked workflow execution threads
    with (
        patch.object(QMessageBox, "critical") as mock_critical,
        patch("adsorpy.gui.show_surface") as mock_show_surface,
    ):
        qtbot.mouseClick(surface_tab.generate_surface_button, Qt.MouseButton.LeftButton)

        # Core rendering loops should be skipped entirely
        mock_show_surface.assert_not_called()
        # Modal dialogue handles should log user errors safely
        mock_critical.assert_called_once_with(surface_tab, "Input Error", "Seed must be a positive integer")


def test_generate_surface_respects_dark_mode_visibility_rules(surface_tab: SurfaceGeneration, qtbot: QtBot) -> None:
    """Verify theme queries check system configuration parameters before passing plotting commands.

    :param surface_tab: SurfaceGeneration widget.
    :param qtbot: Mocks interaction.
    """
    mock_hints = MagicMock()
    mock_hints.colorScheme.return_value = Qt.ColorScheme.Dark

    with (
        patch.object(QGuiApplication, "instance") as mock_app_instance,
        patch("adsorpy.gui.show_surface") as mock_show_surface,
    ):
        mock_app_instance.return_value.styleHints.return_value = mock_hints

        qtbot.mouseClick(surface_tab.generate_surface_button, Qt.MouseButton.LeftButton)

        _, kwargs = mock_show_surface.call_args
        assert kwargs["dark_mode_bool"] is True


def test_generate_surface_finalises_widget_renderer_and_caches_state(surface_tab: SurfaceGeneration) -> None:
    """Verify that generating a surface locks the aspect ratio and saves parameters.

    :param surface_tab: SurfaceGeneration widget.
    """
    fake_svg_bytes = b"<svg><rect width='10' height='10'/></svg>"

    # Pre-set variables to trigger the baseline collection branch
    surface_tab.state.seed_input.setText("")
    surface_tab.lattice_input.setValue(1.0)
    surface_tab.surface_dropdown.setCurrentText("square")
    surface_tab.surface_count = 50

    with (
        patch("adsorpy.gui.show_surface") as mock_show_surface,
        patch.object(surface_tab.svg_widget, "load") as mock_svg_load,
        patch.object(surface_tab.svg_widget, "renderer") as mock_renderer_getter,
    ):
        # Create a mock renderer to check aspect ratio mutations
        mock_renderer = MagicMock()
        mock_renderer_getter.return_value = mock_renderer

        # Inject mock bytes payload when show_surface accesses the BytesIO stream
        def mock_show_surface_impl(*args: P.args, **kwargs: P.kwargs) -> None:  # type: ignore[valid-type] # pyright: ignore[reportUnnecessaryTypeIgnoreComment]
            """Mock function to implement the show surface function.

            :param args: Positional arguments.
            :param kwargs: Keyword arguments.
            """
            kwargs["filepath"].write(fake_svg_bytes)  # pyright: ignore[reportAttributeAccessIssue]

        mock_show_surface.set_defaults_or_side_effect = mock_show_surface_impl
        mock_show_surface.side_effect = mock_show_surface_impl

        surface_tab.generate_surface()

        mock_svg_load.assert_called_once_with(fake_svg_bytes)

        # Assert that the rendering framework explicitly enforced aspect ratios
        mock_renderer.setAspectRatioMode.assert_called_once_with(Qt.AspectRatioMode.KeepAspectRatio)

        # Verify both local components and global states synchronised clean typed dataclasses
        adapter = TypeAdapter(SurfaceParameters)
        adapter.validate_python(surface_tab.state.surface_params)
        assert surface_tab.state.surface_params == surface_tab.stored_params
        assert surface_tab.stored_params["lattice_type"] == "square"  # pyright: ignore[reportOptionalSubscript, reportTypedDictNotRequiredAccess]


def test_error_method_displays_critical_message_box(surface_tab: SurfaceGeneration) -> None:
    """Verify that calling the error convenience hook triggers a critical modal window."""
    error_message_string = "Matrix configuration validation failed."

    # Intercept the static critical dialogue call to prevent test window blocking
    with patch.object(QMessageBox, "critical") as mock_critical:
        surface_tab.error(error_message_string)

        # Verify the modal container registered proper parenting, titles, and payloads
        mock_critical.assert_called_once_with(surface_tab, "Input Error", error_message_string)
