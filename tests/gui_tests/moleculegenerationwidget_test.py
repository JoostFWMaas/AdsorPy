# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Test the MoleculeGeneration class of the `gui.py` module."""

from __future__ import annotations

import inspect
from itertools import count
from typing import Any
from unittest.mock import patch

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QWidget,
)
from pytestqt.qtbot import QtBot

from adsorpy import molecule_lib
from adsorpy.gui import (
    AppState,
    MoleculeGeneration,
    ReorderableListWidget,
    ZoomableSvgWidget,
)


@pytest.fixture
def molecule_tab(qtbot: QtBot) -> MoleculeGeneration:
    """Instantiate the MoleculeGeneration tab using a real AppState context.

    :param qtbot: Qt instance to mock interaction.
    """
    state = AppState()
    tab = MoleculeGeneration(state)
    qtbot.addWidget(tab)
    return tab


def test_initial_structural_layout_states(molecule_tab: MoleculeGeneration) -> None:
    """Verify that layout panels, dropdown configurations, and splitters load correctly.

    :param molecule_tab: MoleculeGeneration widget.
    """
    assert molecule_tab.main_splitter.count() == 3  # noqa: PLR2004
    assert molecule_tab.main_splitter.orientation() == Qt.Orientation.Horizontal

    # # Check default combo box configuration entries
    # expected_items = sorted(["hexagonal", "square", "honeycomb"])
    # actual_items = [molecule_tab.surface_dropdown.itemText(i) for i in range(molecule_tab.surface_dropdown.count())]
    # assert actual_items == expected_items


def test_data_storage_initializes_empty_metrics(molecule_tab: MoleculeGeneration) -> None:
    """Verify list managers and counting sequences initialise cleanly on startup."""
    assert isinstance(molecule_tab.mol_list_counter, count)
    assert next(molecule_tab.mol_list_counter) == 0
    assert molecule_tab.mol_params_list == []
    assert isinstance(molecule_tab.param_widgets, dict)
    assert isinstance(molecule_tab.opt_checkboxes, dict)


def test_panels_assemble_structural_widgets_correctly(molecule_tab: MoleculeGeneration) -> None:
    """Verify three distinct control layouts organise inside expected parent wrappers."""
    # Validate Left Panel components
    assert isinstance(molecule_tab.func_dropdown, QComboBox)
    assert molecule_tab.add_molecule_button.text() == "Add new molecule"

    # Validate Center Panel viewport frame components
    assert isinstance(molecule_tab.svg_widget, ZoomableSvgWidget)
    assert molecule_tab.svg_widget.minimumSize().width() == 600  # noqa: PLR2004

    # Validate Right Panel grid listing components
    assert isinstance(molecule_tab.molecule_list_widget, ReorderableListWidget)
    assert isinstance(molecule_tab.delete_btn, QPushButton)


def test_discover_molecule_generators_filters_library_signatures(
    molecule_tab: MoleculeGeneration,
    subtests: pytest.Subtests,
) -> None:
    """Verify reflection lookup scans module keys, ignoring hidden files and invalid types."""
    generators = molecule_tab._discover_molecule_generators()

    temp_generators = {
        name: func
        for name, func in molecule_lib.__dict__.items()
        if inspect.isfunction(func)
        # and not key.startswith("_")
        # and func.__module__ == molecule_lib.__name__
        # and inspect.signature(func).return_annotation in {"Polygon", "dict[str, str | float | list[str] | None]"}
    }
    # temp_generators = dict(sorted(temp_generators.items()))

    assert temp_generators, "Unfiltered generator list is empty."
    assert generators, "Generator list is empty."
    assert len(temp_generators) > len(generators), "Unfiltered and filtered generator list are identical."

    # Public valid functions must exist inside the discovered keys map
    for key, genv in temp_generators.items():
        filtered: bool = (
            key.startswith("_")
            or genv.__module__ != molecule_lib.__name__
            or inspect.signature(genv).return_annotation not in {"Polygon", "dict[str, str | float | list[str] | None]"}
        )
        fullname = "filtered" if filtered else "exists"
        with subtests.test(key + " " + fullname):
            if filtered:
                assert key not in generators, f"Generator should be filtered out: {key}"
            else:
                assert key in generators, f"Generator should not be filtered out: {key}"
                print(f"Test function module: {genv.__module__}")
                print(f"Discovered function module: {generators[key].__module__}")
                assert genv == generators[key], f"Registered wrong function for: {key}"


def test_delete_previous_layout_clears_widgets_and_nested_layouts(
    molecule_tab: MoleculeGeneration,
    qtbot: QtBot,
) -> None:
    """Verify recursive traversal tears down layout structures without creating memory leaks."""
    # Build a sample mock layout hierarchy inside our parameter layout holder
    # Assert widgets are added and valid
    assert molecule_tab.param_layout.count(), "The molecule parameter layout is empty!"
    # Act: Trigger the deep layout demolition pass
    molecule_tab._delete_previous_layout()
    # Assert: The outer layout framework should be completely flushed clean
    assert molecule_tab.param_layout.count() == 0


def test_build_param_inputs_creates_labeled_grid_elements(molecule_tab: MoleculeGeneration, qtbot: QtBot) -> None:
    """Verify reflection engine extracts arguments to compile type-safe input controls."""

    # Define a custom molecule generation function to inspect
    def temp_generator(
        distance: float,
        ignore_atoms: str | list[str] | None = None,
    ) -> None:
        """Temporary generator function.

        :param distance: Distance.
        :param ignore_atoms: Optional Tag.
        :returns: A polygon of some kind.
        """

    molecule_tab.generators = {"custom_mol": temp_generator}  # pyright: ignore[reportAttributeAccessIssue]
    molecule_tab.func_dropdown.addItems(["custom_mol"])

    # Inject mock parameter document descriptions and global layout helpers
    mock_docs = {"distance": "The spatial span boundary metric."}
    with (
        patch("adsorpy.gui.extract_param_docs", return_value=mock_docs),
        patch(
            "adsorpy.gui.get_type_hints",
            return_value={
                "distance": QDoubleSpinBox,
                "ignore_atoms": QLineEdit,
            },
        ),
        patch("adsorpy.gui._make_horizontal_line", return_value=QWidget()),
        patch.object(molecule_tab, "_build_symmetry_controls") as mock_sym,
        patch.object(molecule_tab, "_build_action_buttons") as mock_act,
    ):
        # Execute form compilation loop
        molecule_tab.build_param_inputs("custom_mol")

        # 1. Assert helper control decorators triggered down the execution pipeline
        mock_sym.assert_called_once()
        mock_act.assert_called_once()

        # 2. Check that active tracking dictionaries populated with correct instances
        assert "distance" in molecule_tab.param_widgets
        assert isinstance(molecule_tab.param_widgets["distance"], QDoubleSpinBox)
        assert "ignore_atoms" in molecule_tab.param_widgets
        assert isinstance(molecule_tab.param_widgets["ignore_atoms"], QLineEdit)

        # 3. Check optional checkbox toggles are managed correctly
        assert "ignore_atoms" in molecule_tab.opt_checkboxes
        checkbox = molecule_tab.opt_checkboxes["ignore_atoms"]
        assert isinstance(checkbox, QCheckBox)
        assert not checkbox.isChecked()
        assert not molecule_tab.param_widgets["ignore_atoms"].isEnabled()


def test_build_bad_param_inputs_raises_critical_dialogue(molecule_tab: MoleculeGeneration, qtbot: QtBot) -> None:
    """Verify reflection engine extracts arguments to compile type-safe input controls."""

    # Define a custom molecule generation function to inspect
    def bad_generator(
        _: float,
    ) -> None:
        """Temporary generator function.

        :param _: Distance.
        :returns: A polygon of some kind.
        """

    molecule_tab.generators = {"custom_mol": bad_generator}  # pyright: ignore[reportAttributeAccessIssue]
    molecule_tab.func_dropdown.addItems(["custom_mol"])

    # Inject mock parameter document descriptions and global layout helpers
    mock_docs = {"distance": "The spatial span boundary metric."}
    with (
        patch("adsorpy.gui.extract_param_docs", return_value=mock_docs),
        patch(
            "adsorpy.gui.get_type_hints",
            return_value={
                "distance": QDoubleSpinBox,
            },
        ),
        patch("adsorpy.gui._make_horizontal_line", return_value=QWidget()),
        patch.object(molecule_tab, "_build_symmetry_controls") as mock_sym,
        patch.object(molecule_tab, "_build_action_buttons") as mock_act,
    ):
        # Execute form compilation loop

        expected_msg = "Parameter input widget mismatch: _ and QDoubleSpinBox"
        with patch.object(QMessageBox, "critical") as mock_critical:
            molecule_tab.build_param_inputs("custom_mol")

            mock_critical.assert_called_once_with(
                molecule_tab,
                "Value Error",
                expected_msg,
            )

        mock_sym.assert_called_once()
        mock_act.assert_called_once()
        assert "_" not in molecule_tab.param_widgets


def test_build_bad_param_inputs_raises_error(molecule_tab: MoleculeGeneration, qtbot: QtBot) -> None:
    """Verify reflection engine extracts arguments to compile type-safe input controls."""

    # Define a custom molecule generation function to inspect
    def bad_generator(  # type: ignore[explicit-any]
        ignore_atoms: Any | None = None,  # noqa: ANN401
    ) -> None:
        """Temporary generator function.

        :param ignore_atoms: Optional Tag.
        :returns: A polygon of some kind.
        """

    molecule_tab.generators = {"custom_mol": bad_generator}  # pyright: ignore[reportAttributeAccessIssue]
    molecule_tab.func_dropdown.addItems(["custom_mol"])

    # Inject mock parameter document descriptions and global layout helpers
    mock_docs = {"distance": "The spatial span boundary metric."}
    with (
        patch("adsorpy.gui.extract_param_docs", return_value=mock_docs),
        patch(
            "adsorpy.gui.get_type_hints",
            return_value={
                "ignore_atoms": QLineEdit,
            },
        ),
        patch("adsorpy.gui._make_horizontal_line", return_value=QWidget()),
        # patch.object(molecule_tab, "_build_symmetry_controls") as mock_sym,
        # patch.object(molecule_tab, "_build_action_buttons") as mock_act,
    ):
        # Execute form compilation loop

        expected_msg = "Unsupported parameter annotation: 'Any | None'."
        with pytest.raises(TypeError, match=expected_msg):
            molecule_tab.build_param_inputs("custom_mol")
