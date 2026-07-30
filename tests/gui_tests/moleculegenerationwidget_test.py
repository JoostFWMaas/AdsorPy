import inspect
import sys
import types
from itertools import count
from unittest.mock import ANY, MagicMock, patch

import pytest
from pydantic import FilePath, ValidationError
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from shapely import Polygon

from src.adsorpy.gui import (
    AppState,
    FilePickerWidget,
    MoleculeGeneration,
    MoleculeParameters,
    ReorderableListWidget,
)

mock_lib_name = "src.adsorpy.molecule_library"
mock_lib = types.ModuleType(mock_lib_name)
mock_lib.__name__ = mock_lib_name


# Add dummy molecule generators with correct signature annotations for reflection
def make_square_molecule() -> Polygon:
    """Generate a dummy square layout."""
    return {}


def make_triangle_molecule() -> Polygon:
    """Generate a dummy triangle layout."""
    return {}


def _private_helper_function():
    """Ignore by reflection."""


mock_lib.make_square_molecule = make_square_molecule
mock_lib.make_triangle_molecule = make_triangle_molecule
mock_lib._private_helper_function = _private_helper_function

sys.modules[mock_lib_name] = mock_lib
# ----------------------------------------------------------------------

# Now import your application widgets safely


# --- Test Stub subclass to isolate missing layout method calls smoothly ---
class TestableMoleculeGeneration(MoleculeGeneration):
    """Subclass of MoleculeGeneration to bypass complex downstream rendering functions."""

    def _delete_previous_layout(self) -> None:
        """Stub out layout demolition to prevent residual init loops from breaking test states."""
        pass

    def build_param_inputs(self, name: str) -> None:
        """Stub parameter generation."""
        self.built_inputs_for = name

    def plot_molecule(self) -> None:
        """Stub plotting loop."""
        self.plot_molecule_called = True

    def show_molecule_settings(self, current, previous) -> None:
        """Stub list widget selection slot."""
        pass

    def sync_list_order(self) -> None:
        """Stub reordering action slot."""
        pass

    def delete_molecule(self) -> None:
        """Stub deletion action slot."""
        pass

    def _assemble_layout(self, left: QWidget, center: QScrollArea, right: QWidget) -> None:
        """Capture panels on self to perform structural placement validations."""
        self.left_panel = left
        self.center_scroll = center
        self.right_panel = right


# --- Fixtures ---


@pytest.fixture
def mock_state():
    """Generate clean isolated application contexts."""
    return MagicMock(spec=AppState)


@pytest.fixture
def molecule_tab(qtbot, mock_state):
    """Instantiate the testable molecule tab component under a mocked library context."""
    # Inject a clean reflection destination patch targeting our setup mock module
    with (
        patch("src.adsorpy.molecule_lib", mock_lib),
        patch("src.adsorpy.gui.MoleculeGeneration._fetch_setting", return_value=0),
    ):
        tab = TestableMoleculeGeneration(mock_state)
        qtbot.addWidget(tab)
        return tab


# ----------------------------------------------------
# 1. Component Lifecycle & Panel Construction Tests
# ----------------------------------------------------


def test_data_storage_initializes_empty_metrics(molecule_tab):
    """Verify list managers and counting sequences initialize cleanly on startup."""
    assert isinstance(molecule_tab.mol_list_counter, count)
    assert next(molecule_tab.mol_list_counter) == 0
    assert molecule_tab.mol_params_list == []
    assert isinstance(molecule_tab.param_widgets, dict)
    assert isinstance(molecule_tab.opt_checkboxes, dict)


def test_panels_assemble_structural_widgets_correctly(molecule_tab):
    """Verify three distinct control layouts organize inside expected parent wrappers."""
    # Validate Left Panel components
    assert isinstance(molecule_tab.func_dropdown, QComboBox)
    assert molecule_tab.add_molecule_button.text() == "Add new molecule"

    # Validate Center Panel viewport frame components
    assert isinstance(molecule_tab.center_scroll, QScrollArea)
    assert molecule_tab.center_scroll.widgetResizable() is True
    assert molecule_tab.svg_widget.minimumSize().width() == 600

    # Validate Right Panel grid listing components
    assert isinstance(molecule_tab.molecule_list_widget, ReorderableListWidget)
    assert isinstance(molecule_tab.delete_btn, QPushButton)


# ----------------------------------------------------
# 2. Metaprogramming & Reflection Discovery Tests
# ----------------------------------------------------


def test_discover_molecule_generators_filters_library_signatures(molecule_tab: TestableMoleculeGeneration) -> None:
    """Verify reflection lookup scans module keys, ignoring hidden files and invalid types.

    :param molecule_tab: Testable version of MoleculeGeneration widget.
    """
    # Execute the reflection discovery against the real loaded library
    generators = molecule_tab._discover_molecule_generators()

    # 1. Assert your actual production functions are successfully discovered
    assert "circulium" in generators
    assert "discorectangle" in generators
    assert "dogbonium" in generators
    assert "polygonium" in generators
    assert "xyz_reader" in generators

    # 2. Verify that internal private layout helpers are correctly filtered out
    # (Assuming your real library has some internal helpers starting with an underscore)
    for name in generators:
        assert not name.startswith("_")


# ----------------------------------------------------
# 3. Dynamic Dropdown Interaction Loop Tests
# ----------------------------------------------------


def test_index_zero_edge_case_initializes_automatically(molecule_tab):
    """Verify starting with standard selection targets builds default parameters instantly."""
    # Initialized inside __init__ because target_index returned 0
    assert molecule_tab.built_inputs_for == molecule_tab.func_dropdown.currentText()
    assert molecule_tab.plot_molecule_called is True


def test_changing_dropdown_selection_triggers_recompilation(molecule_tab, qtbot):
    """Verify switching molecule presets fires reconstruction and refresh hooks automatically."""
    target_molecule = "discorectangle"

    molecule_tab.built_inputs_for = None
    molecule_tab.plot_molecule_called = False

    molecule_tab.func_dropdown.setCurrentText(target_molecule)

    assert molecule_tab.built_inputs_for == target_molecule
    assert molecule_tab.plot_molecule_called is True



# Import the actual classes being targeted

# ----------------------------------------------------
# 1. Recursive Layout Cleanup Assertions
# ----------------------------------------------------


def test_delete_previous_layout_clears_widgets_and_nested_layouts(molecule_tab, qtbot):
    """Verify recursive traversal tears down layout structures without creating memory leaks."""
    # Build a sample mock layout hierarchy inside our parameter layout holder
    grid = QGridLayout()
    dummy_widget = QLabel("Old Parameter Item")
    nested_layout = QHBoxLayout()
    nested_widget = QSpinBox()

    nested_layout.addWidget(nested_widget)
    grid.addWidget(dummy_widget, 0, 0)
    grid.addLayout(nested_layout, 0, 1)
    molecule_tab.param_layout.addLayout(grid)

    # Assert widgets are added and valid
    assert molecule_tab.param_layout.count() == 1

    # Act: Trigger the deep layout demolition pass
    molecule_tab._delete_previous_layout()

    # Assert: The outer layout framework should be completely flushed clean
    assert molecule_tab.param_layout.count() == 0


# ----------------------------------------------------
# 2. Reflection Form Generation & Input Building Tests
# ----------------------------------------------------


import inspect
from unittest.mock import MagicMock, patch
from PySide6.QtWidgets import QDoubleSpinBox, QWidget, QMessageBox, QCheckBox, QLineEdit


def test_build_param_inputs_creates_labeled_grid_elements(molecule_tab, qtbot):
    """Verify reflection engine extracts arguments to compile type-safe input controls."""

    # 1. Target your valid production engine key mapping
    target_function = "circulium"
    molecule_tab.generators = {target_function: lambda length: None}

    # Ensure the dropdown list contains our target item text
    if molecule_tab.func_dropdown.findText(target_function) == -1:
        molecule_tab.func_dropdown.addItems([target_function])

    # 2. Build a native, clean Signature object with exactly ONE simple required parameter
    mock_param = inspect.Parameter(
        name="length",
        kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
        default=inspect.Parameter.empty,  # Is required -> natively skips set_content completely!
        annotation="float",
    )
    mock_sig = MagicMock()
    mock_sig.parameters = {"length": mock_param}

    mock_docs = {"length": "The spatial span boundary metric."}

    # Pass the genuine class type-hint reference
    mock_hints = {"length": QDoubleSpinBox}

    # 3. Use an authentic un-mocked widget layout wrapper to pass native isinstance() evaluations
    def side_effect_widget_factory(annotation, default):
        box = QDoubleSpinBox()
        qtbot.addWidget(box)  # Keep alive within the robot lifecycle framework
        return box

    molecule_tab._create_param_widget = side_effect_widget_factory

    with (
        patch("src.adsorpy.gui.extract_param_docs", return_value=mock_docs),
        patch("src.adsorpy.gui.get_type_hints", return_value=mock_hints),
        patch("src.adsorpy.gui._make_horizontal_line", return_value=QWidget()),
        patch("src.adsorpy.gui.is_valid_param", return_value=True),
        patch("inspect.signature", return_value=mock_sig),
        patch.object(QMessageBox, "critical") as mock_critical,
        patch.object(molecule_tab, "_build_symmetry_controls") as mock_sym,
        patch.object(molecule_tab, "_build_action_buttons") as mock_act,
    ):
        # 4. Execute form compilation loop
        molecule_tab.build_param_inputs(target_function)

        # Confirm no early unhandled mismatch exceptions or missing key errors halted execution
        mock_critical.assert_not_called()

        # 5. CONFIRMED: Execution traverses the entire configuration loop flawlessly
        mock_sym.assert_called_once()
        mock_act.assert_called_once()

        # 6. Verify local tracking components successfully registered the generated widgets
        assert "length" in molecule_tab.param_widgets
        assert isinstance(molecule_tab.param_widgets["length"], QDoubleSpinBox)


def test_build_param_inputs_mismatch_shows_critical_dialog(molecule_tab):
    """Verify widget specification mismatched annotations trigger modal error prompts gracefully."""

    def broken_generator(bad_param: complex):
        pass

    molecule_tab.generators = {"broken_mol": broken_generator}

    with (
        patch("src.adsorpy.gui.extract_param_docs", return_value={}),
        patch("src.adsorpy.gui.get_type_hints", return_value={"bad_param": QSpinBox}),
        patch.object(QMessageBox, "critical") as mock_critical,
    ):
        # Bypassing factory logic tracking via an explicit catch wrapper
        try:
            molecule_tab.build_param_inputs("broken_mol")
        except TypeError:
            pass  # Catching the unmapped fallback exception type error path

        # Ensure errors log output notifications safely rather than allowing clean execution passes
        mock_critical.assert_called()


# ----------------------------------------------------
# 3. Factory Routing Constraint Tests
# ----------------------------------------------------


@pytest.mark.parametrize(
    "annotation, default, expected_class, min_val, max_val",
    [
        ("float", inspect.Parameter.empty, QDoubleSpinBox, -999.0, 999.0),
        ("PositiveFloat", 2.5, QDoubleSpinBox, 0.0001, 999.0),
        ("NonNegativeFloat", inspect.Parameter.empty, QDoubleSpinBox, 0.0, 999.0),
        ("int", inspect.Parameter.empty, QSpinBox, -999, 999),
        ("PositiveInt", inspect.Parameter.empty, QSpinBox, 1, 999),
        ("FilePath", inspect.Parameter.empty, FilePickerWidget, None, None),
        ("str | list[str] | None", inspect.Parameter.empty, QLineEdit, None, None),
    ],
)
def test_create_param_widget_factory_assignments(annotation, default, expected_class, min_val, max_val):
    """Verify that type annotations resolve to the correct specialized interactive widget."""
    widget = MoleculeGeneration._create_param_widget(annotation, default)

    assert isinstance(widget, expected_class)

    # Test numeric boundary ranges if applicable to type
    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        assert widget.minimum() == min_val
        assert widget.maximum() == max_val


def test_create_param_widget_throws_type_error_on_unsupported_annotations():
    """Verify that strange or unmapped typing structures throw standard Python TypeErrors."""
    with pytest.raises(TypeError, match="Unsupported parameter annotation: 'complex'"):
        MoleculeGeneration._create_param_widget("complex", inspect.Parameter.empty)


# ----------------------------------------------------
# 4. Drag and Drop Array Synchronization Tests
# ----------------------------------------------------


def test_sync_list_order_rearranges_data_cache_arrays(molecule_tab):
    """Verify drag-and-drop transformations modify layout lists without corruption."""
    # Pre-populate dummy metadata list array elements
    param_a = {"name": "Molecule Alpha"}
    param_b = {"name": "Molecule Beta"}
    param_c = {"name": "Molecule Gamma"}

    molecule_tab.mol_params_list = [param_a, param_b, param_c]

    # Act: Move Item at Index 0 (Alpha) to Index 2 (End of queue)
    molecule_tab.sync_list_order(old_index=0, new_index=2)

    # Assert structural integrity updates matching user layouts shifts
    assert molecule_tab.mol_params_list == [param_b, param_c, param_a]
    assert molecule_tab.mol_params_list[0] == param_b
    assert molecule_tab.mol_params_list[2] == param_a

def test_sync_list_order_updates_shared_global_state(molecule_tab):
    """Verify dragging items around re-assigns the full list matrix to the global state sync layer."""
    molecule_tab.mol_params_list = [{"name": "A"}, {"name": "B"}]

    # Trigger reordering simulation
    molecule_tab.sync_list_order(old_index=0, new_index=1)

    # Verify both internal cache lists and top-level synced memory states updated completely
    assert molecule_tab.mol_params_list == [{"name": "B"}, {"name": "A"}]
    assert molecule_tab.state.molecule_param_list == [{"name": "B"}, {"name": "A"}]


@pytest.mark.parametrize(
    "is_checked, expected_group, expected_circle",
    [
        (True, "D", "O(2)"),  # Reflection checks toggle Dihedral configurations
        (False, "C", "SO(2)"),  # Default un-checked configurations fall back to Cyclic groupings
    ],
)
def test_update_symmetry_tooltip_mutates_text_labels(molecule_tab, is_checked, expected_group, expected_circle):
    """Verify toggling reflection switches updates rotation tooltips with math notation variants."""
    # Build standard containers manually since we stubbed _build_symmetry_controls earlier
    molecule_tab.rot_sym_label = QLabel()
    molecule_tab.rot_sym_spinbox = QSpinBox()

    molecule_tab._update_symmetry_tooltip(is_checked)

    # Assert that string formatting re-arranged documentation tokens correctly
    assert expected_group in molecule_tab.rot_sym_label.toolTip()
    assert expected_circle in molecule_tab.rot_sym_spinbox.toolTip()


# ----------------------------------------------------
# 2. Form Parameter Dictionary Extraction Tests
# ----------------------------------------------------


def test_get_param_values_ignores_disabled_optional_widgets(molecule_tab):
    """Verify form parsers omit unchecked optional values from final simulation inputs."""
    # Set up form widgets mixed between active inputs and disabled optional widgets
    active_spin = QSpinBox()
    active_spin.setValue(12)

    disabled_line = QLineEdit()
    disabled_line.setText("ignored_text")
    disabled_line.setEnabled(False)  # Simulated un-checked option flag checkbox state

    molecule_tab.param_widgets = {"step_size": active_spin, "custom_name": disabled_line}

    extracted_data = molecule_tab.get_param_values()

    # Active field must exist; disabled container field must be completely ignored
    assert extracted_data["step_size"] == 12
    assert "custom_name" not in extracted_data


# ----------------------------------------------------
# 3. Viewport Render Re-plot & Signal Connection Tests
# ----------------------------------------------------


def test_plot_molecule_skipped_if_checkbox_is_unchecked(molecule_tab):
    """Verify render pipelines skip computation tasks entirely when preview toggles are false."""
    molecule_tab.show_molecule_checkbox = QCheckBox()
    molecule_tab.show_molecule_checkbox.setChecked(False)  # Explicitly disabled canvas flag

    with patch("src.adsorpy.molecule_lib.save_molecule_svg") as mock_save:
        molecule_tab.plot_molecule()
        mock_save.assert_not_called()


def test_plot_molecule_success_updates_svg_canvas_view(molecule_tab):
    """Verify successful plotting passes memory buffers down to the SVG widget pipeline."""
    molecule_tab.show_molecule_checkbox = QCheckBox()
    molecule_tab.show_molecule_checkbox.setChecked(True)

    # Configure dropdown state queries
    molecule_tab.func_dropdown.setCurrentText("make_square_molecule")

    # Mock data extraction method to return standard configurations
    dummy_params = {"length": 5.0}
    molecule_tab.get_param_values = MagicMock(return_value=dummy_params)

    fake_svg = b"<svg>molecule</svg>"

    # Intercept library file writers and widget loaders
    with (
        patch("src.adsorpy.molecule_lib.save_molecule_svg") as mock_save,
        patch.object(molecule_tab.svg_widget, "load") as mock_svg_load,
    ):
        # Side-effect function to copy vector lines into internal test streams
        def mock_save_impl(polygon, filename):
            filename.write(fake_svg)

        mock_save.side_effect = mock_save_impl

        molecule_tab.plot_molecule()

        # Verify plotting function ran using standard form configurations
        mock_save.assert_called_once_with(ANY, filename=ANY)
        mock_svg_load.assert_called_once_with(fake_svg)


def test_plot_molecule_error_redirects_to_dialog(molecule_tab):
    """Verify engine geometry exceptions are caught and pass alerts down to dialog boxes."""
    molecule_tab.show_molecule_checkbox = QCheckBox()
    molecule_tab.show_molecule_checkbox.setChecked(True)
    molecule_tab.get_param_values = MagicMock(return_value={})

    # Force the background generation tool to throw a validation rule error
    with (
        patch("src.adsorpy.molecule_lib.save_molecule_svg", side_effect=ValueError("Invalid atom coordinates")),
        patch.object(molecule_tab, "error") as mock_error,
    ):
        molecule_tab.plot_molecule()

        # Error function hook should catch the exception string output safely
        mock_error.assert_called_once_with("Invalid atom coordinates")


# ----------------------------------------------------
# 4. First Time Loader Interactive Script Tests
# ----------------------------------------------------


def test_launch_first_time_loader_missing_widget_reports_error(molecule_tab):
    """Verify launching file loaders without configuration items logs runtime key errors."""
    molecule_tab.param_widgets = {}  # Missing file_name component index completely

    with patch.object(QMessageBox, "critical") as mock_critical:
        molecule_tab.launch_first_time_loader()
        mock_critical.assert_called_once_with(None, "Key Error", "Parameter file_name not found in widget.")


def test_add_molecule_success_creates_and_caches_parameters(molecule_tab):
    """Verify adding a molecule updates list views, data caches, and application state sync arrays."""
    # 1. Arrange baseline parameters
    molecule_tab.func_dropdown.setCurrentText("make_square_molecule")

    dummy_settings = {"length": 1.25, "file_name": "/home/user/carbon.xyz"}
    molecule_tab.get_param_values = MagicMock(return_value=dummy_settings)

    # Instantiate symmetry UI components to bypass missing object lookups
    molecule_tab.refl_sym_checkbox = QCheckBox()
    molecule_tab.refl_sym_checkbox.setChecked(True)
    molecule_tab.rot_sym_spinbox = QSpinBox()
    molecule_tab.rot_sym_spinbox.setValue(2)
    molecule_tab.rot_cnt_spinbox = QSpinBox()
    molecule_tab.rot_cnt_spinbox.setValue(180)

    # 2. Mock out Pydantic parsing wrapper types to return clean objects
    with patch("src.adsorpy.gui.PydanticPolygon") as mock_pydantic_poly:
        mock_pydantic_poly.return_value = "ValidatedPolygonDataStructure"

        # 3. Act
        molecule_tab.add_molecule()

        # 4. Assert row tracker listings added the new tag entry cleanly
        assert molecule_tab.molecule_list_widget.count() == 1
        assert molecule_tab.molecule_list_widget.item(0).text() == "carbon.xyz #0"

        # Assert internal cache array successfully populated a structured MoleculeParameters instance
        assert len(molecule_tab.mol_params_list) == 1
        created_record = molecule_tab.mol_params_list[0]

        assert isinstance(created_record, dict)
        assert created_record.index == 0
        assert created_record.function_name == "make_square_molecule"
        assert created_record.label == "carbon.xyz #0"
        assert created_record.refl_sym is True
        assert created_record.rot_sym == 2
        assert created_record.rot_cnt == 180

        # Assert application shared context array received the synchronized mutation pass
        assert molecule_tab.state.molecule_param_list == molecule_tab.mol_params_list
        assert "Added: carbon.xyz" in molecule_tab.output_label.text()


def test_add_molecule_validation_error_routes_to_dialog(molecule_tab):
    """Verify that schema constraints validation errors stop insertions and trigger modal boxes."""
    molecule_tab.func_dropdown.setCurrentText("make_square_molecule")
    molecule_tab.get_param_values = MagicMock(return_value={})

    # Force generator function call trigger to intentionally raise a structural Pydantic validation error
    mock_func = MagicMock(side_effect=ValidationError.from_exception_data(title="Value Error", line_errors=[]))
    molecule_tab.generators["make_square_molecule"] = mock_func

    with patch.object(molecule_tab, "error") as mock_error:
        molecule_tab.add_molecule()

        # Verification pipelines must reject caching actions and dispatch text alerts
        mock_error.assert_called_once()
        assert len(molecule_tab.mol_params_list) == 0


# ----------------------------------------------------
# 2. Record Removal & Selection Clearance Tests
# ----------------------------------------------------


def test_delete_molecule_flushes_row_caches_and_resets_selection(molecule_tab):
    """Verify clicking delete drops dictionary tracking elements and clears highlight blocks."""
    # Pre-populate list elements
    molecule_tab.molecule_list_widget.addItem("molecule_a #0")
    molecule_tab.molecule_list_widget.addItem("molecule_b #1")
    molecule_tab.mol_params_list = [{"label": "a"}, {"label": "b"}]

    # Simulate clicking and highlighting the second row item (index 1)
    molecule_tab.molecule_list_widget.setCurrentRow(1)

    molecule_tab.delete_molecule()

    # Assert records dropped matching array keys mapping sequences
    assert molecule_tab.molecule_list_widget.count() == 1
    assert len(molecule_tab.mol_params_list) == 1
    assert "Molecule deleted" in molecule_tab.output_label.text()

    # Highlighting focuses must clear out completely
    assert molecule_tab.molecule_list_widget.currentRow() == -1


def test_delete_molecule_safely_ignores_negative_indices(molecule_tab):
    """Verify that running delete calls without highlighted rows exits instantly without errors."""
    molecule_tab.mol_params_list = [{"label": "persistent_item"}]
    molecule_tab.molecule_list_widget.setCurrentRow(-1)  # No active highlighted target

    molecule_tab.delete_molecule()

    # Cache lists must remain un-mutated
    assert len(molecule_tab.mol_params_list) == 1


# ----------------------------------------------------
# 3. Form History Re-hydration Tests
# ----------------------------------------------------


def test_show_molecule_settings_populates_widgets_from_history(molecule_tab):
    """Verify selecting a row item extracts cached properties back into input text fields."""
    # Build a simulated history profile using dictionary lookups
    cached_record = {
        "function_name": "make_square_molecule",
        "settings": {"atom_distance": 2.50, "element_symbol": "H"},
    }
    molecule_tab.mol_params_list = [cached_record]
    molecule_tab.molecule_list_widget.addItem("square #0")
    molecule_tab.molecule_list_widget.setCurrentRow(0)

    # Prepare mock parameter input destinations
    mock_spin = QDoubleSpinBox()
    mock_line = QLineEdit()
    molecule_tab.param_widgets = {"atom_distance": mock_spin, "element_symbol": mock_line}

    # Configure option checkboxes flags targets
    mock_check = QCheckBox()
    molecule_tab.opt_checkboxes = {"atom_distance": mock_check}

    # Intercept system validators and form update setters
    with (
        patch("src.adsorpy.gui.is_valid_param", return_value=True),
        patch("src.adsorpy.gui.set_content") as mock_set_content,
    ):
        # Execute history rendering re-hydration sequence
        molecule_tab.show_molecule_settings()

        # Verify setter helpers triggered to push float/string data back into layout inputs
        assert mock_set_content.call_count == 2
        mock_set_content.assert_any_call(mock_spin, 2.50)
        mock_set_content.assert_any_call(mock_line, "H")

        # Optional checkbox flags should react and snap to Checked true conditions
        assert mock_check.isChecked() is True
        assert molecule_tab.show_molecule_checkbox.isChecked() is True
