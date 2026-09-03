# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Test the MoleculeGeneration class of the `gui.py` module."""

from __future__ import annotations

import inspect
from functools import partial
from itertools import count
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import pytest
from _pytest.monkeypatch import MonkeyPatch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import BaseModel, ValidationError
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpacerItem,
    QSpinBox,
    QWidget,
)
from pytestqt.qtbot import QtBot

from adsorpy import molecule_lib
from adsorpy.gui import (
    AppState,
    FilePickerWidget,
    MoleculeGeneration,
    ReorderableListWidget,
    ZoomableSvgWidget,
)


def make_isolated_tab(qtbot: QtBot, monkeypatch: MonkeyPatch) -> MoleculeGeneration:
    """Create a totally fresh tab using prefilled pytest hooks.

    :param qtbot: Simulate user input.
    :param monkeypatch: Pytest fixture to mock parameters.
    :returns: MoleculeGeneration tab instance.
    """
    state = AppState()
    tab = MoleculeGeneration(state)
    qtbot.addWidget(tab)
    monkeypatch.setattr(tab, "plot_molecule", Mock())
    return tab


@pytest.fixture
def molecule_tab_factory(qtbot: QtBot, monkeypatch: MonkeyPatch) -> partial[MoleculeGeneration]:
    """Create factory function for molecule tab.

    :param qtbot: Simulate user input.
    :param monkeypatch: Pytest fixture to mock parameters.
    :returns: Partial MoleculeGeneration tab instance.
    """
    return partial(make_isolated_tab, qtbot=qtbot, monkeypatch=monkeypatch)


def test_initial_structural_layout_states(molecule_tab_factory: partial[MoleculeGeneration]) -> None:
    """Verify that layout panels, dropdown configurations, and splitters load correctly.

    :param molecule_tab: MoleculeGeneration widget.
    """
    molecule_tab = molecule_tab_factory()

    assert molecule_tab.main_splitter.count() == 3  # noqa: PLR2004
    assert molecule_tab.main_splitter.orientation() == Qt.Orientation.Horizontal


def test_data_storage_initialises_empty_metrics(molecule_tab_factory: partial[MoleculeGeneration]) -> None:
    """Verify list managers and counting sequences initialise cleanly on startup."""
    molecule_tab = molecule_tab_factory()
    assert isinstance(molecule_tab.mol_list_counter, count)
    assert next(molecule_tab.mol_list_counter) == 0
    assert molecule_tab.mol_params_list == []
    assert isinstance(molecule_tab.param_widgets, dict)
    assert isinstance(molecule_tab.opt_checkboxes, dict)


def test_panels_assemble_structural_widgets_correctly(molecule_tab_factory: partial[MoleculeGeneration]) -> None:
    """Verify three distinct control layouts organise inside expected parent wrappers."""
    molecule_tab = molecule_tab_factory()
    # Validate Left Panel components
    assert isinstance(molecule_tab.func_dropdown, QComboBox)

    # Validate Center Panel viewport frame components
    assert isinstance(molecule_tab.svg_widget, ZoomableSvgWidget)
    assert molecule_tab.svg_widget.minimumSize().width() == 600  # noqa: PLR2004

    # Validate Right Panel grid listing components
    assert isinstance(molecule_tab.molecule_list_widget, ReorderableListWidget)
    assert isinstance(molecule_tab.delete_btn, QPushButton)


def test_discover_molecule_generators_filters_library_signatures(
    molecule_tab_factory: partial[MoleculeGeneration],
    subtests: pytest.Subtests,
) -> None:
    """Verify reflection lookup scans module keys, ignoring hidden files and invalid types."""
    molecule_tab = molecule_tab_factory()
    generators = molecule_tab._discover_molecule_generators()

    temp_generators = {name: func for name, func in molecule_lib.__dict__.items() if inspect.isfunction(func)}

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
    molecule_tab_factory: partial[MoleculeGeneration],
) -> None:
    """Verify recursive traversal tears down layout structures without creating memory leaks."""
    molecule_tab = molecule_tab_factory()
    # Build a sample mock layout hierarchy inside our parameter layout holder
    # Assert widgets are added and valid
    assert molecule_tab.param_layout.count(), "The molecule parameter layout is empty."
    # Act: Trigger the deep layout demolition pass
    molecule_tab._delete_previous_layout()
    # Assert: The outer layout framework should be completely flushed clean
    assert molecule_tab.param_layout.count() == 0


def test_build_param_inputs_creates_labeled_grid_elements(
    molecule_tab_factory: partial[MoleculeGeneration],
    monkeypatch: MonkeyPatch,
) -> None:
    """Verify reflection engine extracts arguments to compile type-safe input controls."""
    molecule_tab = molecule_tab_factory()

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

    temp_mol_name = "custom_mol"

    molecule_tab.generators = {temp_mol_name: temp_generator}  # pyright: ignore[reportAttributeAccessIssue]
    molecule_tab.func_dropdown.addItems([temp_mol_name])

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
        molecule_tab.build_param_inputs(temp_mol_name)

        mock_sym.assert_called_once()
        mock_act.assert_called_once()

        assert "distance" in molecule_tab.param_widgets
        assert isinstance(molecule_tab.param_widgets["distance"], QDoubleSpinBox)
        assert "ignore_atoms" in molecule_tab.param_widgets
        assert isinstance(molecule_tab.param_widgets["ignore_atoms"], QLineEdit)

        assert "ignore_atoms" in molecule_tab.opt_checkboxes
        checkbox = molecule_tab.opt_checkboxes["ignore_atoms"]
        assert isinstance(checkbox, QCheckBox)
        assert not checkbox.isChecked()
        assert not molecule_tab.param_widgets["ignore_atoms"].isEnabled()


def test_build_bad_param_inputs_raises_critical_dialogue(molecule_tab_factory: partial[MoleculeGeneration]) -> None:
    """Verify reflection engine extracts arguments to compile type-safe input controls."""
    molecule_tab = molecule_tab_factory()

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


def test_build_param_inputs_trips_future_annotation_check(
    molecule_tab_factory: partial[MoleculeGeneration],
    monkeypatch: MonkeyPatch,
) -> None:
    """Simulate a library function that is missing the __future__ annotations import to verify it suggests the fix."""
    molecule_tab = molecule_tab_factory()

    def bad_library_func(_: None) -> None:
        """Do nothing and have no annotation."""

    molecule_tab.generators["bad_func"] = bad_library_func  # pyright: ignore[reportArgumentType]

    mock_param = MagicMock(spec=inspect.Parameter)
    mock_param.default = inspect.Parameter.empty

    type(mock_param).annotation = PropertyMock(return_value=int)

    mock_signature = MagicMock(spec=inspect.Signature)
    type(mock_signature).parameters = PropertyMock(return_value={"sample_arg": mock_param})

    monkeypatch.setattr(inspect, "signature", Mock(return_value=mock_signature))

    monkeypatch.setattr("adsorpy.gui.extract_param_docs", Mock(return_value={}))
    monkeypatch.setattr("adsorpy.gui.get_type_hints", Mock(return_value={}))
    mock_error = Mock()
    monkeypatch.setattr(MoleculeGeneration, "error", mock_error)

    molecule_tab.build_param_inputs("bad_func")

    mock_error.assert_called_once_with(
        "Parameter is not a string. Use ``from __future__ import annotations`` to ensure this.",
    )


def test_build_bad_param_inputs_raises_error(molecule_tab_factory: partial[MoleculeGeneration]) -> None:
    """Verify reflection engine extracts arguments to compile type-safe input controls."""
    molecule_tab = molecule_tab_factory()

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
    ):
        # Execute form compilation loop

        expected_msg = "Unsupported parameter annotation: 'Any | None'."
        with pytest.raises(TypeError, match=expected_msg):
            molecule_tab.build_param_inputs("custom_mol")


def generate_pydantic_error() -> ValidationError:
    """Generate a pydantic validation error.

    :returns: ValidationError.
    :raises AssertionError: If a ValidationError is not returned.
    """

    class MockModel(BaseModel):
        must_be_int: int

    try:
        MockModel(must_be_int="not_an_int")  # type: ignore[arg-type]
    except ValidationError as e:
        return e
    errmsg = "Failed to generate a ValidationError context."
    raise AssertionError(errmsg)


def test_add_molecule_success(molecule_tab_factory: partial[MoleculeGeneration], monkeypatch: MonkeyPatch) -> None:
    """Verify that a successful molecule generation updates UI components and tracking state."""
    molecule_tab = molecule_tab_factory()
    molecule_name = "dogbonium"
    molecule_tab.func_dropdown.setCurrentText(molecule_name)
    refl_flag = True
    molecule_tab.refl_sym.setChecked(refl_flag)
    rot_sym = 4
    molecule_tab.rot_sym.setValue(rot_sym)

    molecule_tab.mol_list_counter = count(start=1)
    molecule_tab.mol_params_list = []

    molecule_tab.add_molecule()

    assert molecule_name in molecule_tab.output_label.text()
    assert molecule_tab.molecule_list_widget.count() == 1
    assert molecule_tab.molecule_list_widget.item(0).text() == f"{molecule_name} #1"

    assert len(molecule_tab.mol_params_list) == 1
    stored_param = molecule_tab.mol_params_list[0]
    assert stored_param["index"] == 1
    assert stored_param["function_name"] == molecule_name
    assert stored_param["label"] == f"{molecule_name} #1"
    assert stored_param["refl_sym"] is refl_flag
    assert stored_param["rot_sym"] == rot_sym
    assert molecule_tab.state.molecule_param_list == molecule_tab.mol_params_list


def test_add_molecule_validation_error(
    molecule_tab_factory: partial[MoleculeGeneration],
    monkeypatch: MonkeyPatch,
) -> None:
    """Verify that a Pydantic ValidationError stops execution and bubbles via the error tracker."""
    molecule_tab = molecule_tab_factory()
    molecule_name = "xyz_reader"
    molecule_tab.func_dropdown.setCurrentText(molecule_name)
    molecule_tab.molecule_list_widget.clear()
    molecule_tab.mol_params_list = []

    expected_error = generate_pydantic_error()
    mock_generator_func = Mock(side_effect=expected_error)
    molecule_tab.generators = {molecule_name: mock_generator_func}  # pyright: ignore[reportAttributeAccessIssue]

    mock_error_channel = Mock()
    monkeypatch.setattr(molecule_tab, "get_param_values", Mock(return_value={}))
    monkeypatch.setattr(molecule_tab, "error", mock_error_channel)

    molecule_tab.add_molecule()

    mock_generator_func.assert_called_once()
    mock_error_channel.assert_called_once_with(str(expected_error))

    assert molecule_tab.molecule_list_widget.count() == 0
    assert not len(molecule_tab.mol_params_list)
    assert molecule_tab.output_label.text() == ""


@pytest.mark.parametrize(
    ("dropdown_text", "expected_func_key"),
    [
        ("first_time_loader", "xyz_reader"),
        ("xyz_reader", "xyz_reader"),
    ],
)
def test_add_molecule_fallback_routing(
    molecule_tab_factory: partial[MoleculeGeneration],
    monkeypatch: MonkeyPatch,
    dropdown_text: str,
    expected_func_key: str,
) -> None:
    """Verify that special generator naming triggers appropriate fallback routing aliases."""
    molecule_tab = molecule_tab_factory()
    molecule_tab.func_dropdown.setCurrentText(dropdown_text)
    molecule_tab.mol_list_counter = count(start=1)

    mock_generator_func = Mock(return_value={})
    molecule_tab.generators = {expected_func_key: mock_generator_func}  # pyright: ignore[reportAttributeAccessIssue]

    monkeypatch.setattr(molecule_tab, "get_param_values", Mock(return_value={}))

    molecule_tab.add_molecule()

    mock_generator_func.assert_called_once()


def test_delete_molecule_success(molecule_tab_factory: partial[MoleculeGeneration], monkeypatch: MonkeyPatch) -> None:
    """Verify that deleting a selected molecule updates UI elements and tracking lists."""
    molecule_tab = molecule_tab_factory()

    name1 = "molecule_a #1"
    name2 = "molecule_b #2"

    molecule_tab.molecule_list_widget.addItem(name1)
    molecule_tab.molecule_list_widget.addItem(name2)

    # Mirror items in the tracking data structures
    param_mock_a = Mock()
    param_mock_b = Mock()

    monkeypatch.setattr(molecule_tab, "show_molecule_settings", Mock())

    molecule_tab.mol_params_list = [param_mock_a, param_mock_b]
    molecule_tab.state.molecule_param_list = molecule_tab.mol_params_list

    molecule_tab.molecule_list_widget.setCurrentRow(1)
    assert molecule_tab.molecule_list_widget.currentRow() == 1

    molecule_tab.delete_molecule()

    # Verify tracking collections have been purged of the accurate item index
    assert len(molecule_tab.mol_params_list) == 1
    assert molecule_tab.mol_params_list[0] is param_mock_a
    assert molecule_tab.state.molecule_param_list == [param_mock_a]

    # Verify UI tracking list component was updated
    assert molecule_tab.molecule_list_widget.count() == 1
    assert molecule_tab.molecule_list_widget.item(0).text() == name1

    # Verify status label messages and selection state clearances
    assert molecule_tab.output_label.text() == "Molecule deleted"
    assert molecule_tab.molecule_list_widget.currentRow() == -1


def test_delete_molecule_no_selection_returns_early(molecule_tab_factory: partial[MoleculeGeneration]) -> None:
    """Verify that trying to delete a molecule when nothing is selected returns immediately."""
    molecule_tab = molecule_tab_factory()

    molecule_tab.molecule_list_widget.addItem("molecule_a #1")

    param_mock = Mock()
    molecule_tab.mol_params_list = [param_mock]
    molecule_tab.state.molecule_param_list = molecule_tab.mol_params_list
    molecule_tab.output_label.setText("Initial State")

    # Set selection state explicitly to -1 (nothing selected)
    molecule_tab.molecule_list_widget.setCurrentRow(-1)

    molecule_tab.delete_molecule()

    # Verify data layers remain completely untouched
    assert len(molecule_tab.mol_params_list) == 1
    assert molecule_tab.state.molecule_param_list == [param_mock]

    # Verify UI items and notification labels did not shift
    assert molecule_tab.molecule_list_widget.count() == 1
    assert molecule_tab.output_label.text() == "Initial State"


def test_launch_first_time_loader_without_name(
    molecule_tab_factory: partial[MoleculeGeneration],
    monkeypatch: MonkeyPatch,
) -> None:
    """Test whether this function initialises correctly."""
    molecule_tab = molecule_tab_factory()
    mock_picker = MagicMock()
    mock_picker.text.return_value = ""
    molecule_tab.show_molecule_flag = False

    monkeypatch.setitem(molecule_tab.param_widgets, "file_name", mock_picker)

    monkeypatch.setattr(
        molecule_lib,
        "first_time_loader",
        Mock(side_effect=AssertionError("Task failed successfully.")),
    )

    with pytest.raises(AssertionError, match=r"Task failed successfully."):
        molecule_tab.launch_first_time_loader()

    mock_picker.text.assert_called()
    mock_picker.browse_button.click.assert_called_once()

    assert molecule_tab.show_molecule_flag is False, "Molecule flag should remain set to False."


def test_launch_first_time_loader_fully(
    molecule_tab_factory: partial[MoleculeGeneration],
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test that launch_first_time_loader runs completely when a valid file is provided."""
    molecule_tab = molecule_tab_factory()
    temp_file = tmp_path / "molecule.xyz"
    temp_file.write_text("dummy data")

    mock_picker = MagicMock()
    mock_picker.text.return_value = str(temp_file)
    monkeypatch.setitem(molecule_tab.param_widgets, "file_name", mock_picker)

    mock_output = {"x_offset": 1.5, "roll": 45.0}
    mock_loader = Mock(return_value=mock_output)
    monkeypatch.setattr(molecule_lib, "first_time_loader", mock_loader)

    monkeypatch.setattr("adsorpy.gui.is_valid_param", Mock(return_value=True))
    mock_set_content = Mock()
    monkeypatch.setattr("adsorpy.gui.set_content", mock_set_content)

    molecule_tab.opt_checkboxes = {"x_offset": MagicMock(), "roll": MagicMock()}

    molecule_tab.show_molecule_flag = False

    molecule_tab.launch_first_time_loader()

    mock_loader.assert_called_once_with(temp_file)

    assert mock_set_content.call_count == len(mock_output)
    input_flag = True
    molecule_tab.opt_checkboxes["x_offset"].setChecked.assert_called_once_with(input_flag)  # pyright: ignore[reportAttributeAccessIssue]
    molecule_tab.opt_checkboxes["roll"].setChecked.assert_called_once_with(input_flag)  # pyright: ignore[reportAttributeAccessIssue]

    assert molecule_tab.show_molecule_flag is True


def test_launch_first_time_loader_missing_key_error(
    molecule_tab_factory: partial[MoleculeGeneration],
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that a critical error occurs if 'file_name' is missing from param_widgets."""
    molecule_tab = molecule_tab_factory()
    molecule_tab.param_widgets = {}

    mock_critical = Mock()
    monkeypatch.setattr(QMessageBox, "critical", mock_critical)

    molecule_tab.launch_first_time_loader()

    mock_critical.assert_called_once_with(molecule_tab, "Key Error", "Parameter file_name not found in widget.")
    assert not hasattr(molecule_tab, "show_molecule_flag") or molecule_tab.show_molecule_flag is False


def test_launch_first_time_loader_invalid_param_error(
    molecule_tab_factory: partial[MoleculeGeneration],
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that a critical error occurs if the loader returns an invalid key."""
    molecule_tab = molecule_tab_factory()
    mock_picker = MagicMock()
    mock_picker.text.return_value = "/mock/path.xyz"
    monkeypatch.setitem(molecule_tab.param_widgets, "file_name", mock_picker)

    mock_loader = Mock(return_value={"invalid_key_name": "some_value"})
    monkeypatch.setattr(molecule_lib, "first_time_loader", mock_loader)

    monkeypatch.setattr("adsorpy.gui.is_valid_param", Mock(return_value=False))
    mock_critical = Mock()
    monkeypatch.setattr(QMessageBox, "critical", mock_critical)

    if "invalid_key_name" in molecule_tab.param_widgets:
        del molecule_tab.param_widgets["invalid_key_name"]

    molecule_tab.launch_first_time_loader()

    mock_critical.assert_called_once_with(molecule_tab, "Key Error", "Not a valid key: invalid_key_name")
    assert not hasattr(molecule_tab, "show_molecule_flag") or molecule_tab.show_molecule_flag is False


@given(st.data())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_sync_list_order_with_hypothesis(
    molecule_tab_factory: partial[MoleculeGeneration],
    data: st.DataObject,
) -> None:
    """Property-based test ensuring list sync properties hold true across arbitrary sequences."""
    # 1. Generate an arbitrary list of unique strings (at least 1 item long)
    # We filter out empty/falsy strings to ensure the 'if taken_item:' condition passes
    molecule_tab = molecule_tab_factory()
    initial_list = data.draw(st.lists(st.text(min_size=1), min_size=1, unique=True))

    # 2. Dynamically draw old and new indices that are guaranteed to be within bounds
    max_idx = len(initial_list) - 1
    old_index = data.draw(st.integers(min_value=0, max_value=max_idx))
    new_index = data.draw(st.integers(min_value=0, max_value=max_idx))

    # Track the element we expect to move
    target_item = initial_list[old_index]

    # 3. Setup mock environment
    molecule_tab.mol_params_list = initial_list.copy()  # pyright: ignore[reportAttributeAccessIssue]
    molecule_tab.state = MagicMock()

    # 4. Execute the function
    molecule_tab.sync_list_order(old_index, new_index)

    # 5. Assert the invariant properties
    # Property A: The item must now exist at the exact new destination index
    assert molecule_tab.mol_params_list[new_index] == target_item

    # Property B: The length of the list must remain identical
    assert len(molecule_tab.mol_params_list) == len(initial_list)

    # Property C: The set of items must not have changed (order changed, contents didn't)
    assert set(molecule_tab.mol_params_list) == set(initial_list)

    # Property D: State must sync correctly with the list tracking
    assert molecule_tab.state.molecule_param_list == molecule_tab.mol_params_list


def test_build_left_panel_with_target_index_0(
    molecule_tab_factory: partial[MoleculeGeneration],
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that the ``build_param_inputs`` function is called when the target index is 0."""
    molecule_tab = molecule_tab_factory()
    mock_build_param_inputs = Mock()
    mock_settings = Mock(return_value=0)
    monkeypatch.setattr(MoleculeGeneration, "build_param_inputs", mock_build_param_inputs)
    monkeypatch.setattr(MoleculeGeneration, "_fetch_setting", mock_settings)

    molecule_tab._build_left_panel()
    mock_build_param_inputs.assert_called_once()


def test_delete_previous_layout(molecule_tab_factory: partial[MoleculeGeneration], monkeypatch: MonkeyPatch) -> None:
    """Test that the ``delete_previous_layout`` function deletes everything correctly."""
    molecule_tab = molecule_tab_factory()
    assert molecule_tab.param_layout.count(), "Layout should not start empty."
    molecule_tab._delete_previous_layout()
    assert not molecule_tab.param_layout.count(), "Layout should have been cleared."

    molecule_tab.param_layout.addItem(QSpacerItem(1, 1))
    assert molecule_tab.param_layout.count(), "Layout should not start empty."
    molecule_tab._delete_previous_layout()
    assert not molecule_tab.param_layout.count(), "Layout should have been cleared."


def test_create_param_widget(
    molecule_tab_factory: partial[MoleculeGeneration], monkeypatch: MonkeyPatch, subtests: pytest.Subtests,
) -> None:
    """Verify that specific type annotations result in correct behaviour."""
    molecule_tab = molecule_tab_factory()

    all_param_types = {
        "float": (0.0, QDoubleSpinBox),
        "PositiveFloat": (1.0, QDoubleSpinBox),
        "NonNegativeFloat": (2.0, QDoubleSpinBox),
        "float | None": (3.0, QDoubleSpinBox),
        "int": (4, QSpinBox),
        "PositiveInt": (5, QSpinBox),
        "FilePath": ("6", FilePickerWidget),
        "str | list[str] | None": ("7", QLineEdit),
    }

    for param_input, (param_default, param_type) in all_param_types.items():
        with subtests.test(param_name=param_input):
            assert isinstance(molecule_tab._create_param_widget(param_input, param_default), param_type)


def test_create_param_widget_error(
    molecule_tab_factory: partial[MoleculeGeneration], monkeypatch: MonkeyPatch, subtests: pytest.Subtests,
) -> None:
    """Verify that incorrect type annotations result in a raised TypeError."""
    molecule_tab = molecule_tab_factory()
    annotation = "InvalidType"
    with pytest.raises(TypeError, match=f"Unsupported parameter annotation: '{annotation}'."):
        molecule_tab._create_param_widget(annotation, "")


def test_get_param_values(
    molecule_tab_factory: partial[MoleculeGeneration], monkeypatch: MonkeyPatch, subtests: pytest.Subtests,
) -> None:
    """Test whether the get param values function works correctly."""
    molecule_tab = molecule_tab_factory()
    disabled_widget = QLineEdit()
    disabled_widget.setDisabled(True)
    spinbox = QSpinBox()
    value = 1
    spinbox.setValue(value)
    lineedit = QLineEdit()
    text = "success"
    lineedit.setText(text)
    input_dict = {
        "spinbox": spinbox,
        "lineedit": lineedit,
        "disabled_widget": disabled_widget,
    }

    monkeypatch.setattr(molecule_tab, "param_widgets", input_dict)
    output = molecule_tab.get_param_values()

    assert "spinbox" in output
    assert "lineedit" in output
    assert "disabled_widget" not in output
    assert output["spinbox"] == value
    assert output["lineedit"] == text


def test_error(molecule_tab_factory: partial[MoleculeGeneration], monkeypatch: MonkeyPatch) -> None:
    """Test whether the error function works correctly."""
    molecule_tab = molecule_tab_factory()
    mock_error = Mock()
    input_message = "Test message"
    monkeypatch.setattr(QMessageBox, "critical", mock_error)

    molecule_tab.error(input_message)

    mock_error.assert_called_once_with(molecule_tab, "Input Error", input_message)
