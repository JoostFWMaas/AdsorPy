# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""GUI module of adsorpy."""  # TODO: Make a new repo for this!

from __future__ import annotations

import inspect
import io
import json
import multiprocessing
import re
import sys
import textwrap
import webbrowser
from collections import defaultdict
from dataclasses import field

import matplotlib as mpl

if sys.version_info >= (3, 11):
    from datetime import UTC, datetime  # For datetime stamping and seed generation.
else:
    from datetime import datetime

from itertools import count
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Generic,
    Literal,
    ParamSpec,
    Self,
    TypeVar,
    cast,
    get_origin,
    override,
)

import dask
import numpy as np
import pydantic
from dask.distributed import Client, as_completed
from pydantic import (
    BeforeValidator,
    ConfigDict,
    NonNegativeInt,
    PlainSerializer,
    PositiveFloat,
    PositiveInt,
    TypeAdapter,
    ValidationError,
)
from PySide6.QtCore import (
    Property,
    QObject,
    QRegularExpression,
    QRunnable,
    QSettings,
    Qt,
    QThreadPool,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QAction,
    QDoubleValidator,
    QDropEvent,
    QGuiApplication,
    QIcon,
    QIntValidator,
    QRegularExpressionValidator,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from shapely import Polygon, from_geojson
from shapely.geometry import mapping

from adsorpy import __version__
from src.adsorpy import molecule_lib
from src.adsorpy.run_simulation import run_simulation, show_surface

T_qobj = TypeVar("T_qobj", bound=QObject)
T_co = TypeVar("T_co", bound=bool | int | str | float, covariant=True)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.random import Generator

    from adsorpy.randomsequentialadsorption import Simulator
    from adsorpy.types import DistArray, IdxArray

    P = ParamSpec("P")
    P_mol = ParamSpec("P_mol", bound=int | float | str | list[str] | None)  # Helps with static type checkers.

def extract_param_docs(func: Callable) -> dict[str, str]:
    """Extract parameters and their types from the docstring of a function.

    This function is written for reStructuredText (rst) style docstrings.

    :param func: The function from which the docstring is extracted.
    :return: The dictionary of parameters and their types (as strings).
    :raises ValueError: If the function has no docstring.
    """
    doc: str | None = inspect.getdoc(func)
    if doc is None:
        errmsg: str = f"Docstring of {func.__name__} is not defined."
        raise ValueError(errmsg)

    param_docs: dict[str, str] = {}
    lines: list[str] = doc.splitlines()

    current_param: str | None = None
    buffer: list[str] = []

    for line in lines:
        param_match = re.match(r":param\s+(\w+)\s*:\s*(.*)", line)
        end_match = re.match(r":returns?:*", line)
        if param_match:
            if current_param and buffer:
                param_docs[current_param] = " ".join(buffer).strip()

            current_param = param_match.group(1)
            buffer = [param_match.group(2).strip()]

        elif end_match:
            break

        elif current_param:
            buffer.append(line.strip())

    # Save last param
    if current_param and buffer:
        param_docs[current_param] = " ".join(buffer).strip()

    return param_docs

def validate_polygon(pol: Polygon | str | dict[str, str | list[tuple[float, float]]]) -> Polygon:
    """Convert the GeoJSON dict data into a real Shapely Polygon or pass the data if it is already a Polygon.

    :param pol: Polygon or GeoJSON format.
    :returns: Polygon.
    :raises TypeError: if the type cannot be converted to Polygon.
    """
    if isinstance(pol, Polygon):
        return pol

    if isinstance(pol, dict):
        # Convert the python dictionary to a valid JSON string first
        json_str = json.dumps(pol)
        return cast("Polygon", from_geojson(json_str))

    if isinstance(pol, str):
        # Turns {"type": "Polygon", "coordinates": ...} into a Shapely object
        return cast("Polygon", from_geojson(pol))

    errmsg =  f"Cannot convert {type(pol)} to a Shapely Polygon"
    raise TypeError(errmsg)

# Define the custom Pydantic-safe type
PydanticPolygon = Annotated[
    Polygon,
    BeforeValidator(validate_polygon),  # This runs before the Pydantic validation step.
    PlainSerializer(mapping, return_type=dict),  # This runs when serialising: standard GeoJSON format
]

@pydantic.dataclasses.dataclass(slots=True, frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class MoleculeParameters:
    """Molecule parameters dataclass.

    :ivar index: Index of the molecule parameters configuration.
    :ivar label: Label of the molecule parameters configuration, guaranteed to be unique.
    :ivar function_name: Function name of the molecule.
    :ivar refl_sym: Reflection symmetry.
    :ivar rot_sym: Rotation symmetry.
    :ivar rot_cnt: Rotation count (before accounting for reflection/rotation symmetry).
    :ivar polygon: 2D polygon representation of the molecule.
    :ivar settings: Function input of the molecule. Defaults to an empty dictionary.
    """

    index: NonNegativeInt
    label: str
    function_name: str
    refl_sym: bool
    rot_sym: NonNegativeInt
    rot_cnt: PositiveInt
    polygon: PydanticPolygon
    settings: dict[str, float | int | str | list[str] | None] = field(default_factory=dict)

pydantic.dataclasses.rebuild_dataclass(MoleculeParameters)


@pydantic.dataclasses.dataclass(slots=True, frozen=True)
class SurfaceParameters:
    """Surface parameters dataclass.

    :ivar lattice_type: Surface lattice type.
    :ivar site_count: Site count of the surface.
    :ivar lattice_a: Lattice spacing of the surface.
    """

    lattice_type: Literal["hexagonal", "triangular", "honeycomb", "square"]
    site_count: PositiveInt
    lattice_a: PositiveFloat


@pydantic.dataclasses.dataclass(slots=True, frozen=True)
class MiscParameters:
    """Miscellaneous parameters dataclass.

    :ivar seed: RNG seed.
    :ivar timestep_limit: Maximum allowed step count of the simulation.
    """

    seed: NonNegativeInt | None = None
    timestep_limit: NonNegativeInt | None = None


class ZoomableSvgWidget(QSvgWidget):
    """SVG widget with zoom capabilities."""

    def __init__(self, parent: QSvgWidget | None = None) -> None:
        """Initialise the ZoomableSvgWidget.

        :param parent: The parent QSvgWidget object. If none is provided, create one of fixed size.
        """
        super().__init__(parent)

        self.zoom_factor = 1.15
        # Set a baseline size so it doesn't collapse to 0x0
        if parent is None:
            self.setMinimumSize(600,600)

    @override
    def wheelEvent(self, event: QWheelEvent) -> None:
        """Override scroll wheel events to support Zoom and Horizontal Pan.

        :param event: The QWheelEvent object, triggered when using the scroll wheel.
        """
        modifiers = event.modifiers()

        # Walk up the layout tree to reliably find the QScrollArea
        # (Necessary since the widget is now inside a centered layout container)
        scroll_area = self.parent()
        while scroll_area and not isinstance(scroll_area, QScrollArea):
            scroll_area = scroll_area.parent()

        # -------------------------------------------------------------
        # CASE 1: CTRL + SCROLL = ZOOM TO MOUSE CURSOR
        # -------------------------------------------------------------
        if modifiers == Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            scale = self.zoom_factor if delta > 0 else 1.0 / self.zoom_factor

            old_size = self.size()
            new_size = old_size * scale
            min_scale: int = 100
            max_scale: int = 50000

            if min_scale < new_size.width() < max_scale:
                mouse_pos_widget = event.position()
                self.setFixedSize(new_size)

                if scroll_area:
                    h_bar = scroll_area.horizontalScrollBar()
                    v_bar = scroll_area.verticalScrollBar()

                    delta_x = mouse_pos_widget.x() * scale - mouse_pos_widget.x()
                    delta_y = mouse_pos_widget.y() * scale - mouse_pos_widget.y()

                    h_bar.setValue(int(h_bar.value() + delta_x))
                    v_bar.setValue(int(v_bar.value() + delta_y))

            event.accept()

        # -------------------------------------------------------------
        # CASE 2: SHIFT + SCROLL = HORIZONTAL PANNING
        # -------------------------------------------------------------
        elif modifiers == Qt.KeyboardModifier.ShiftModifier:
            if scroll_area:
                h_bar = scroll_area.horizontalScrollBar()
                # Determine scroll steps (usually multiples of 120)
                steps = event.angleDelta().y()
                # Shift the horizontal slider position directly
                h_bar.setValue(h_bar.value() - steps)
            event.accept()

        # -------------------------------------------------------------
        # CASE 3: NO MODIFIERS = STANDARD VERTICAL PANNING
        # -------------------------------------------------------------
        elif scroll_area:
            # Safely forward the wheel context to the native viewport
            QApplication.sendEvent(scroll_area.viewport(), event)
        else:
            event.ignore()


class AutoStateMeta(type(QObject), Generic[T_qobj]):
    """Metaclass for AppState to automatically communicate between tabs.

    This metaclass scans the ``fields`` class variable and dynamically
    generates a private storage field (``_field``), a public property
    (``field``), and a Qt notification signal (``fieldChanged``) for each entry.

    :cvar fields: A dictionary mapping state field names to their expected types.
    """

    fields: ClassVar[dict[str, type]]

    def __new__(cls: type[Self], name: str, bases: tuple[type, ...], attrs: dict[str, Any]) -> type[Self]:
        """Create an AutoState class instance.

        :param name: The name of the class.
        :param bases: The base classes.
        :param attrs: The class attributes.
        :return: The AutoState class instance.
        """
        annotations = attrs.get("__annotations__", {})

        # 2. Filter out 'fields' or any other ClassVar/private attributes
        fields = {k: v for k, v in annotations.items() if k != "fields" and not k.startswith("_")}

        for field_name, field_type in fields.items():
            signal_name = f"{field_name}Changed"
            private_name = f"_{field_name}"

            qt_compatible_type = get_origin(field_type) or field_type
            if not isinstance(qt_compatible_type, type) or qt_compatible_type.__module__ != "builtins":
                qt_compatible_type = object
            attrs[signal_name] = Signal(qt_compatible_type)

            def getter(self: Self, private_name: str = private_name) -> object:
                """Get the value.

                :param private_name: private name.
                :returns: the getattr() object.
                """
                return getattr(self, private_name)

            def setter(
                self: Self,
                value: str,
                private_name: str = private_name,
                signal_name: str = signal_name,
            ) -> None:
                """Set the value.

                :param self: the class.
                :param value: the value to update.
                :param private_name: private name.
                :param signal_name: signal name.
                """
                setattr(self, private_name, value)
                getattr(self, signal_name).emit(value)

            attrs[field_name] = property(getter, setter)

        new_class = super().__new__(cls, name, bases, attrs)
        new_class.fields = fields

        return new_class

    def __call__(cls: type[Self], *args: P_mol.args, **kwargs: P_mol.kwargs) -> T_qobj:
        """Instantiate the class and auto-initialise its fields.

        :param args: Positional arguments.
        :param kwargs: Keyword arguments.
        :returns: The initialised class instance.
        """
        obj: T_qobj = super().__call__(*args, **kwargs)

        # Auto-initialise private fields
        for field_name in cls.fields:
            private_name = f"_{field_name}"
            if not hasattr(obj, private_name):
                setattr(obj, private_name, None)

        return obj


class AppState(QObject, metaclass=AutoStateMeta):
    """AppState class to communicate between tabs.

    This class maintains synchronised states across the user interface. Changes
    to any property automatically emit a corresponding ``<property>Changed`` signal.

    :ivar seed_input: The Qt input widget holding the seed value.
    :ivar step_limit: The maximum allowable processing steps.
    :ivar misc_params: Miscellaneous parameters.
    :ivar molecule_param_list: Settings of the molecule(s).
    :ivar surface_params: Settings of the surface.
    :ivar coverages: Coverage of simulation results.
    :ivar fraction_of_covered_area: Fraction of covered area of simulation results.
    :ivar gap_size_distribution: Gap size distribution of simulation results.
    """

    seed_input: QLineEdit
    step_limit: QSpinBox
    misc_params: MiscParameters
    molecule_param_list: list[MoleculeParameters]
    surface_params: SurfaceParameters
    coverages: tuple[DistArray]
    fraction_of_covered_area: tuple[DistArray]
    gap_size_distribution: DistArray


class AdsorpyGUI(QMainWindow):
    """Main window application shell for the AdsorPy simulation engine framework.

    Coordinates the primary window frame, top level configuration menu bars,
    and hooks up the shared global data state across tab layout frames.

    :cvar window_resized:  Signal of (width, height) emitted when the main application window dimensions are modified.
    """

    window_resized: Signal = Signal(int, int)

    def __init__(self) -> None:
        """Initialise frame parameters, global context caches, and child windows."""
        super().__init__()

        self.setWindowTitle("AdsorPy Simulation GUI")

        self.state = AppState()
        """Shared application runtime cache synchronised across all view frames."""

        self._settings = QSettings(type(self).__name__)
        """Persistent platform configuration handle cached between user runtime sessions."""

        # Delegate initialisation to helper workflows
        self._init_menu_bar()
        self._init_tabs()

    def _init_menu_bar(self) -> None:
        """Construct the top level application drop-down menu navigation bars."""
        menubar = self.menuBar()

        # ----------------------------------------------------
        # File Menu Segment
        # ----------------------------------------------------
        file_menu = menubar.addMenu("File")

        self._new_action = QAction(QIcon.fromTheme(QIcon.ThemeIcon.DocumentNew), "New Simulation", self)
        """Action trigger to wipe runtime matrices and reset configurations."""
        self._new_action.triggered.connect(self._reset_app)

        self._open_action = QAction(QIcon.fromTheme(QIcon.ThemeIcon.DocumentOpen), "Open…", self)
        """Action trigger to parse a serialized system JSON file from disk."""
        self._open_action.triggered.connect(self._load_settings_json)

        self._save_action = QAction(QIcon.fromTheme(QIcon.ThemeIcon.DocumentSave), "Save", self)
        """Action trigger to serialize current system conditions to disk."""
        self._save_action.triggered.connect(self._save_settings_json)

        self._exit_action = QAction(QIcon("assets/door-open-out.png"), "Exit", self)
        """Action trigger to safely kill background workers and close window frames."""
        self._exit_action.triggered.connect(self.close)

        file_menu.addAction(self._new_action)
        file_menu.addAction(self._open_action)
        file_menu.addAction(self._save_action)
        file_menu.addSeparator()
        file_menu.addAction(self._exit_action)

        # ----------------------------------------------------
        # Help Menu Segment
        # ----------------------------------------------------
        help_menu = menubar.addMenu("Help")

        self._doc_action = QAction(QIcon("assets/book-open-list.png"), "Documentation (web)", self)
        """Action link opening the official Sphinx HTML documentation site."""
        self._doc_action.triggered.connect(
            lambda: webbrowser.open("https://joostfwmaas.github.io/AdsorPy/"),
        )

        self._wiki_action = QAction(QIcon.fromTheme(QIcon.ThemeIcon.HelpFaq), "Wiki (web)", self)
        """Action link opening the community GitHub developer reference portal."""
        self._wiki_action.triggered.connect(
            lambda: webbrowser.open("https://github.com/JoostFWMaas/AdsorPy/wiki"),
        )

        self._bug_action = QAction(QIcon("assets/bug--exclamation.png"), "Report bug (web)", self)
        """Action link navigating directly to the open project issue dashboard."""
        self._bug_action.triggered.connect(
            lambda: webbrowser.open("https://github.com/JoostFWMaas/AdsorPy/issues"),
        )

        help_menu.addAction(self._doc_action)
        help_menu.addAction(self._wiki_action)
        help_menu.addAction(self._bug_action)

    def _init_tabs(self) -> None:
        """Assemble the central tab frame layout and register sub-dashboards."""
        self.tabs = QTabWidget()
        """Primary navigation container organizing distinct module windows."""

        # Instantiate separate view models sharing the single source of truth state
        self.tabs.addTab(GeneralSettings(self.state), "General")
        self.tabs.addTab(SurfaceGeneration(self.state), "Surface")
        self.tabs.addTab(MoleculeGeneration(self.state), "Molecule(s)")

        self.setCentralWidget(self.tabs)

    def _save_settings_json(self) -> None:
        """Save settings to JSON file."""
        # Validate seed
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Settings",
            self._fetch_setting("last_visited_directory", default=""),
            "JSON Files (*.json);;All Files (*)",
        )
        if not file_path:
            return

        seed_text = self.state.seed_input.text().strip()
        step_limit_val = self.state.step_limit.value()
        # Convert empty fields to None, or parse them to integers
        seed_val = int(seed_text) if seed_text else None
        misc_settings = MiscParameters(seed=seed_val, timestep_limit=step_limit_val)
        surf_settings: SurfaceParameters = self.state.surface_params
        molecule_settings: list[MoleculeParameters] = self.state.molecule_param_list

        try:
            misc_adapter = TypeAdapter(MiscParameters)
            surf_adapter = TypeAdapter(SurfaceParameters)
            mol_adapter = TypeAdapter(list[MoleculeParameters])

            misc_dump = misc_adapter.dump_python(misc_settings)
            surf_dump = surf_adapter.dump_python(surf_settings)
            mol_dump = mol_adapter.dump_python(molecule_settings)

            misc_adapter.validate_python(misc_settings)
            surf_adapter.validate_python(surf_settings)
            mol_adapter.validate_python(molecule_settings)

        except ValidationError as e:
            QMessageBox.critical(
                self,
                "Error Saving File",
                f"Failed to save settings. Structure or type constraints were broken:\n{e}",
            )
            return


        combined_data = {
            "adsorpy_version": __version__,
            "miscellaneous_parameters": misc_dump,
            "surface_parameters": surf_dump,
            "molecule_parameters": mol_dump,
        }

        with Path(file_path).open("w", encoding="utf-8") as f:
            json.dump(combined_data, f, indent=4)

        QMessageBox.information(
            self, "Save Successful", "Your simulation configuration settings have been successfully saved!",
        )

    def _load_settings_json(self) -> None:
        """Load, validate, and version-check simulation settings profiles."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Settings",
            self._fetch_setting("last_visited_directory", default=""),
            "JSON Files (*.json);;All Files (*)",
        )
        if not file_path:
            return  # User cancelled the file selection dialogue

        with Path(file_path).open("rb") as f:
            json_bytes = f.read()
        try:
            raw_structure = TypeAdapter(dict).validate_json(json_bytes)

            misc_adapter = TypeAdapter(MiscParameters)
            surf_adapter = TypeAdapter(SurfaceParameters)
            mol_adapter = TypeAdapter(list[MoleculeParameters])

            # Validate and re-hydrate fields directly into application state
            self.state.misc_params = misc_adapter.validate_python(raw_structure["miscellaneous_parameters"])
            self.state.surface_params = surf_adapter.validate_python(raw_structure["surface_parameters"])
            self.state.molecule_param_list = mol_adapter.validate_python(raw_structure["molecule_parameters"])

            # Synchronise GUI with the newly loaded state
            misc = self.state.misc_params
            self.state.seed_input.setText(str(misc.seed) if misc.seed is not None else "")

            # self.log("Settings successfully loaded and validated.")

        except (KeyError, ValidationError) as e:
            QMessageBox.critical(
                self,
                "Error Loading File",
                f"Failed to parse settings file. Structure or type constraints were broken:\n{e}",
            )

    def _fetch_setting(self, name: str, default: T_co, return_type: type[T_co] | None = None) -> T_co:
        """Fetch settings by checking if they exist followed by their value.

        :param name: The name of the setting to fetch.
        :param default: The default value to return if the setting does not exist.
        :param return_type: The default return type if the setting exists. If not given, type(default) is used.
        :returns: The setting value if it exists, or else the default.
        """
        check_type = type(default) if return_type is None else return_type
        return cast("T_co", self._settings.value(name, defaultValue=default, type=check_type))

    def _reset_app(self) -> None:
        """Close and open the window to reset the app safely and completely."""
        app = QApplication.instance()

        self.hide()
        new_window = AdsorpyGUI()
        app.activeWindow = new_window
        new_window.resize(1600, 900)
        new_window.show()
        self.deleteLater()

    @override  # This decorator is used to indicate a method overrides a method of the base class.
    def resizeEvent(self, event: QResizeEvent) -> None:
        """Trigger automatically whenever the window size changes.

        :param event: QResizeEvent, an event changing the window size.
        """
        # Get the new size from the event object
        new_size = event.size()
        width: int = new_size.width()
        height: int = new_size.height()

        self.window_resized.emit(width, height)

        super().resizeEvent(event)


class GeneralSettings(QWidget):
    """General simulation configuration dashboard tab view.

    Provides inputs for setting the execution step boundaries, absolute pseudo-random
    number generator seeds, and renders real-time structural vector tracking maps.
    """

    def __init__(self, state: AppState) -> None:
        """Initialise validation engines and build structural control modules.

        :param state: AppState object for communication between tab widgets.
        """
        super().__init__()  # Inherit from the super() class (in this case: AppState).

        self._settings = QSettings(type(self).__name__)
        """Persistent platform configuration handle cached between user runtime sessions."""

        self.state = state
        """App state object for communication between tab widgets."""

        # Run UI Initialisation steps
        self._init_validators()

        # Extract widgets/layouts from the initialisation helpers
        # (Assuming _init_controls sets up fields like seed, step limits, etc.)
        controls_layout = self._init_controls()
        svg_widget = self._init_svg_view()

        # Create the Left Panel: Wrap controls layout inside a clean container QWidget
        left_panel = QWidget()
        left_panel.setLayout(controls_layout)

        # Create the Centre Panel: Wrap SVG view in a QScrollArea for responsiveness
        centre_scroll = QScrollArea()
        centre_scroll.setWidgetResizable(True)
        centre_scroll.setWidget(svg_widget)

        self.state.surface_paramsChanged.connect(self._on_surface_changed)
        self.state.molecule_param_listChanged.connect(self._on_molecules_changed)

        # Clean up scroll area borders to integrate smoothly with the splitter look
        # centre_scroll.setFrameShape(QScrollArea.FrameShape.NoFrame)

        # Create the Right Panel: Create a panel for listing generated arrays/molecules
        # right_panel = self._init_management_panel() # Or: right_panel = QWidget()
        # right_panel = QWidget()

        # Unify sub-panels using exact splitter framework layout method
        self._assemble_layout(left=left_panel, center=centre_scroll)

    def _fetch_setting(self, name: str, default: T_co, return_type: type[T_co] | None = None) -> T_co:
        """Fetch settings by checking if they exist followed by their value.

        :param name: The name of the setting to fetch.
        :param default: The default value to return if the setting does not exist.
        :param return_type: The default return type if the setting exists. If not given, type(default) is used.
        :returns: The setting value if it exists, or else the default.
        """
        check_type: type[T_co] = type(default) if return_type is None else return_type
        return cast("T_co", self._settings.value(name, defaultValue=default, type=check_type))

    def _init_validators(self) -> None:
        """Instantiate validation models for text constraint processing."""
        self._seed_validator = QRegularExpressionValidator(regularExpression=QRegularExpression(r"^\d+$"))
        """Restricts string parameters strictly to absolute positive digits."""

    def _init_controls(self) -> QVBoxLayout:
        """Assemble environment settings selectors and connect state triggers.

        :return: A populated vertical layout holding runtime widgets.
        """
        layout = QVBoxLayout()

        # Pseudo-random generator state seed tracking
        layout.addWidget(QLabel("Optional Seed (positive int):"))
        self.seed_input = QLineEdit()
        """Input widget capturing custom random generation bounds."""
        self.seed_input.setValidator(self._seed_validator)
        self.seed_input.setPlaceholderText("e.g. 23")
        self.seed_input.setToolTip("RNG seed for the simulation. If empty, defaults to datetime in microseconds.")
        layout.addWidget(self.seed_input)

        # Sync the specific text field reference directly to global state tracking
        self.state.seed_input = self.seed_input

        # Absolute execution cycle step limit constraints
        layout.addWidget(QLabel("Step limit (optional, > 0 int):"))
        self.step_limit = QSpinBox()
        """Input widget restricting maximum sequential process cycles."""
        self.step_limit.setToolTip("The maximum step limit of the simulation. Stops when done or when limit reached.")
        # self.step_limit.setPlaceholderText("e.g. 1")
        # self.step_limit.setValidator(self._gt_one_validator)
        self.step_limit.setMinimum(0)
        self.step_limit.setMaximum(100000000)
        self.step_limit.setValue(self.get_run_sim_default("timestep_limit"))
        self.step_limit.setAccelerated(True)
        self.step_limit.setStepType(QSpinBox.StepType.AdaptiveDecimalStepType)
        self.state.step_limit = self.step_limit
        layout.addWidget(self.step_limit)

        layout.addWidget(_make_horizontal_line())

        layout.addLayout(self._init_feedback_textboxes())

        self.run_group = QGroupBox()
        """Simulation run group box."""
        run_grid = QGridLayout(self.run_group)

        self.run_button = QPushButton("Run Simulation (1x)")
        """Trigger execution wrapper for adsorpy run."""
        self.run_button.setToolTip("Runs the random sequential adsorption simulation.")
        run_grid.addWidget(self.run_button, 0, 1)
        self.run_button.clicked.connect(self.run_simulation)

        self.repeat_count = QSpinBox()
        """Repeat count value."""
        self.repeat_count.setToolTip("Number of times to repeat. 100 is plenty for most purposes.")
        self.repeat_count.setMinimum(1)
        self.repeat_count.setMaximum(100000)
        self.repeat_count.setValue(self._fetch_setting("repeat_count", 10))
        self.repeat_count.setAccelerated(True)
        self.repeat_count.valueChanged.connect(self._change_bulk_run_value)
        run_grid.addWidget(self.repeat_count, 1, 0)

        self.bulk_run_button = QPushButton(f"Bulk Run ({self.repeat_count.value()}x)")
        """Trigger execution wrapper for adsorpy bulk run."""
        self.bulk_run_button.setToolTip("Runs the simulation multiple times in parallel.")
        self.bulk_run_button.clicked.connect(self.run_batch_simulation)
        run_grid.addWidget(self.bulk_run_button, 1, 1)

        layout.addWidget(self.run_group)

        layout.addWidget(_make_horizontal_line())

        self.progress_bar = QProgressBar()
        """Progress bar for simulations."""
        self.progress_bar.setToolTip("Simulation progress. 'Are we there yet?'")
        self.progress_bar.setRange(0, 100)  # Maps perfectly to percentages (0 to 100)
        self.progress_bar.setValue(0)  # Start empty
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar, stretch=1, alignment=Qt.AlignmentFlag.AlignTop)

        self.coverage_label = QLabel("")
        """Coverage value."""
        self.coverage_label.setVisible(False)
        self.coverage_label.setToolTip("Fraction of surface sites consumed by molecules.")
        layout.addWidget(self.coverage_label, alignment=Qt.AlignmentFlag.AlignTop)
        self.covered_area_label = QLabel("")
        """Fraction of covered area value."""
        self.coverage_label.setToolTip("Fraction of surface area covered by molecule footprints.")
        self.covered_area_label.setVisible(False)
        layout.addWidget(self.covered_area_label, alignment=Qt.AlignmentFlag.AlignTop)

        self.export_results_button = QPushButton("Export Results")
        """Button to export the results."""
        self.export_results_button.setToolTip("Export results by format of choice.")
        self.export_results_button.setVisible(False)
        self.export_results_button.clicked.connect(self.export_results)
        layout.addWidget(self.export_results_button, alignment=Qt.AlignmentFlag.AlignTop)

        layout.addStretch()

        return layout

    def _change_bulk_run_value(self, run_count: int) -> None:
        """Change the bulk run button tooltip.

        :param run_count: The number of times to repeat the simulation.
        """
        self.bulk_run_button.setText(f"Bulk Run ({run_count}x)")
        self._settings.setValue("repeat_count", run_count)

    def _init_feedback_textboxes(self) -> QGridLayout:
        """Provide text to show the user whether data has been loaded."""
        grid_layout = QGridLayout()
        self.initiated_surface_label = QLabel("Surface:")
        """Surface label."""
        self.initiated_surface_textbox = QLabel("Default.")
        """What kind of surface has been loaded."""
        self.initiated_molecules_label = QLabel("Molecule(s):")
        """Molecule label."""
        self.initiated_molecules_textbox = QLabel("Default.")
        """How many molecules has been loaded."""

        grid_layout.addWidget(self.initiated_surface_label, 0, 0)
        grid_layout.addWidget(self.initiated_surface_textbox, 0, 1)
        grid_layout.addWidget(self.initiated_molecules_label, 1, 0)
        grid_layout.addWidget(self.initiated_molecules_textbox, 1, 1)

        return grid_layout

    def _on_surface_changed(self, params: SurfaceParameters | None) -> None:
        """Fire instantly when surface_params changes in another tab."""
        if params is not None:
            self.initiated_surface_textbox.setText("User-defined.")
        else:
            self.initiated_surface_textbox.setText("Default.")

    def _on_molecules_changed(self, mol_list: list[MoleculeParameters] | None) -> None:
        """Fire instantly when molecule_param_list changes in another tab."""
        if mol_list:  # Checks if list exists and is not empty
            count: int = len(mol_list)
            # Create a summary string from the list items if desired
            self.initiated_molecules_textbox.setText(f"{count} molecule{'s'*bool(count - 1)} defined by user.")
        else:
            self.initiated_molecules_textbox.setText("Default.")

    def _init_svg_view(self) -> QSvgWidget:
        """Construct the graphics frame and isolate structural canvas layouts.

        :return: An isolated vector viewport container canvas.
        """
        self.svg_widget = ZoomableSvgWidget()
        """Custom render context displaying loaded vector data files."""
        self.svg_widget.renderer().setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)

        return self.svg_widget

    def _assemble_layout(self, left: QWidget, center: QScrollArea) -> None:
        """Unify sub-panels inside the scalable horizontal splitter framework."""
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        """Main splitter to dynamically divide the window."""

        self.main_splitter.addWidget(left)
        self.main_splitter.addWidget(center)

        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 3)

        root_layout = QVBoxLayout(self)
        root_layout.addWidget(self.main_splitter)
        self.setLayout(root_layout)  # Formally registers root_layout to this QWidget


    @staticmethod
    def get_run_sim_default(name: str) -> str | int | float | None:
        """Get the default value of a function.

        :param name: Name of the parameter.
        :returns: Default value of the parameter.
        :raises ValueError: If the parameter has no default value.
        :raises KeyError: If the parameter does not exist.
        """
        sig: inspect.Signature = inspect.signature(run_simulation)
        param: inspect.Parameter = sig.parameters[name]
        if param.default is inspect.Parameter.empty:
            errmsg: str = f"{name} has no default"
            raise ValueError(errmsg)
        return param.default

    def _prepare_simulation_inputs(self) -> dict[str, Any] | None:
        """Validate UI components and format into a unified dictionary for the simulation engine.

        :returns: Dict as input for run_simulation if successful, None if validation fails.
        """
        seed_text = self.state.seed_input.text().strip()
        step_limit_val = self.state.step_limit.value()

        # Generate seed from current datetime in microseconds if field is empty
        if seed_text:
            seed_val = int(seed_text)
        else:
            how_late = datetime.now(UTC) if sys.version_info >= (3, 11) else datetime.utcnow()  # pyright: ignore[reportPossiblyUnboundVariable, reportDeprecated]
            seed_val = int(how_late.strftime("%Y%m%d%H%M%S%f"))

        try:
            misc_params = MiscParameters(seed=seed_val, timestep_limit=step_limit_val)
        except ValidationError as e:
            errmsg = f"Invalid parameters provided:\n{e}"
            self.error(errmsg)
            return None

        misc_adapter = TypeAdapter(MiscParameters)
        misc_params = misc_adapter.dump_python(misc_params)

        surf_settings: SurfaceParameters = self.state.surface_params
        surf_adapter = TypeAdapter(SurfaceParameters)

        molecule_settings: list[MoleculeParameters] = self.state.molecule_param_list
        mol_adapter = TypeAdapter(MoleculeParameters)

        defaultdict_of_lists = defaultdict(list)
        molecule_settings = [] if molecule_settings is None else molecule_settings

        for dict_in_list in molecule_settings:
            checked_dict = mol_adapter.dump_python(dict_in_list)
            for key, value in checked_dict.items():
                defaultdict_of_lists[key].append(value)

        key_to_fix = "polygon"
        if key_to_fix in defaultdict_of_lists:
            defaultdict_of_lists[key_to_fix] = [
                validate_polygon(geo_item) for geo_item in defaultdict_of_lists[key_to_fix]
            ]

        def replace_keys(
            dict_with_old_keys: defaultdict[str, list[Polygon] | list[int]],
        ) -> dict[str, list[Polygon] | list[int]]:
            old_keys: list[str] = ["polygon", "refl_sym", "rot_sym", "rot_cnt"]
            new_keys: list[str] = ["molecules_list", "reflection_symmetries", "rotation_symmetries", "rotation_counts"]
            mapping: dict[str, str] = dict(zip(old_keys, new_keys, strict=True))
            dict_with_new_keys = dict_with_old_keys.copy()

            for old, new in mapping.items():
                if old in dict_with_new_keys:
                    dict_with_new_keys[new] = dict_with_new_keys.pop(old)

            return dict_with_new_keys

        def filter_dict_for_func(
            data_dict: dict[str, Any], filter_func: Callable[..., Any] = run_simulation,
        ) -> dict[str, Any]:
            sig = inspect.signature(filter_func)
            valid_keys = sig.parameters.keys()
            return {k: v for k, v in data_dict.items() if k in valid_keys}

        dict_of_lists = replace_keys(defaultdict_of_lists)
        dict_of_lists = filter_dict_for_func(dict_of_lists)

        surface_settings: dict[str, int | float | str] = surf_adapter.dump_python(surf_settings)
        surface_settings = {} if surface_settings is None else surface_settings

        return {**misc_params, **surface_settings, **dict_of_lists}

    def run_simulation(self) -> None:
        """Run exactly one instance of the simulation engine."""
        inputs = self._prepare_simulation_inputs()
        if inputs is None:
            return

        self.run_group.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        # self.run_group.setText("Computing...")

        task = BackgroundTask(run_simulation, **inputs)
        task.signals.finished.connect(self._on_simulation_complete)
        task.signals.error.connect(self._on_simulation_error)
        QThreadPool.globalInstance().start(task)

    def run_batch_simulation(self) -> None:
        """Run N parallel instances using Dask with safe child-spawned seeds."""
        n_instances = self.repeat_count.value()
        inputs = self._prepare_simulation_inputs()
        if inputs is None:
            return

        self.run_group.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        def execute_dask_batch(base_inputs: dict[str, Any], total_runs: int, task_ref: Any = None) -> list[Any]:

            tasks = []
            parent_seed: int = base_inputs.get("seed")

            child_seeds = np.random.SeedSequence(parent_seed).spawn(total_runs)

            def wrap_run_func(**kwargs: P.kwargs) -> tuple[DistArray, DistArray, DistArray]:
                output = run_simulation(**kwargs)[-1]
                return output.coverage, output.fraction_of_covered_area, output.analyse_gap_size()

            for seed in child_seeds:
                run_inputs = base_inputs.copy()
                run_inputs["seed"] = seed.generate_state(n_words=1, dtype=np.uint32)[0]
                tasks.append(dask.delayed(wrap_run_func)(**run_inputs))

            workers = max(1, multiprocessing.cpu_count() - 1)

            with Client(n_workers=workers, threads_per_worker=1, processes=True) as client:
                # 1. Asynchronously submit to cluster (returns Futures immediately)
                futures = client.compute(tasks)

                # 2. Iterate dynamically as tasks complete to push progress updates
                for idx, future in enumerate(as_completed(futures), start=1):
                    if task_ref and hasattr(task_ref, "signals"):
                        percentage = int((idx / total_runs) * 100)
                        task_ref.signals.progress.emit(percentage)

                # 3. Pull calculated data back over the network before closing the client context
                results = client.gather(futures)

            return list(results)

        # Instantiate task and pass 'task' itself into the execution function so it can access signals
        task = BackgroundTask(execute_dask_batch, base_inputs=inputs, total_runs=n_instances)
        task.kwargs["task_ref"] = task  # Dynamically inject the task reference into kwargs

        # Connect signals
        if hasattr(self, "progress_bar"):
            task.signals.progress.connect(self.progress_bar.setValue)

        task.signals.finished.connect(self._on_batch_simulation_complete)
        task.signals.error.connect(self._on_simulation_error)
        QThreadPool.globalInstance().start(task)

    def _on_simulation_complete(
        self,
        simulation_outputs: tuple[list[int], DistArray, int | Generator, tuple[IdxArray, ...], IdxArray, Simulator],
    ) -> None:
        """Run lightweight plotting pipeline back on the UI thread."""
        try:
            self.progress_bar.setValue(100)
            output = simulation_outputs[-1]

            # Generate SVG elements in memory
            svg_buffer = io.BytesIO()
            dark_mode_bool: bool = QGuiApplication.instance().styleHints().colorScheme() == Qt.ColorScheme.Dark
            output.svgplot_covered_grid(filename=svg_buffer, dark_mode_bool=dark_mode_bool)
            svg_data = svg_buffer.getvalue()

            # Render UI updates directly
            self.svg_widget.load(svg_data)
            self.svg_widget.renderer().setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)

            # Populate numeric output displays
            self.state.coverages = tuple(output.coverage)
            self.coverage_label.setText(f"Coverage: {np.sum(output.coverage):.4f}")
            self.coverage_label.setToolTip("""Fraction of surface sites consumed/covered by molecules.""")

            self.state.fraction_of_covered_area = tuple(output.fraction_of_covered_area)
            frac_of_covered_area = np.sum(output.fraction_of_covered_area)

            self.state.gap_size_distribution = output.analyse_gap_size()

            self.covered_area_label.setText(f"Fraction of covered area: {frac_of_covered_area:.4f}")
            self.covered_area_label.setToolTip("""Fraction of surface area covered by molecules.""")

        except (ValueError, TypeError, ValidationError) as e:
            self.error(f"Error preparing visual plot data:\n{e}")
        finally:
            # Always unlock button when processing completes
            self.run_group.setEnabled(True)
            self.progress_bar.setVisible(False)
            # self.run_group.setText("Run Simulation")

    def _on_batch_simulation_complete(self, batch_outputs: list[tuple]) -> None:
        """Process multiple parallel output tuples sent back from the dask pool cluster."""
        try:
            if not batch_outputs:
                return

            coverages, fraction_of_covered_area, gapsize_dist = zip(*batch_outputs, strict=True)
            import matplotlib.pyplot as plt
            import seaborn as sns
            mpl.use("Agg")
            fig = plt.figure(figsize=(8, 6))
            gs = fig.add_gridspec(2, 2)

            coverages_arr = np.array(coverages)
            fraction_arr = np.array(fraction_of_covered_area)

            missing_coverage = 1.0 - np.sum(coverages_arr, axis=1)
            missing_fraction = 1.0 - np.sum(fraction_arr, axis=1)

            coverages_final = np.column_stack((coverages_arr, missing_coverage)).tolist()
            fraction_final = np.column_stack((fraction_arr, missing_fraction)).tolist()
            coverages = [np.mean(x) for x in zip(*coverages_final, strict=True)]
            fraction_of_covered_area = [np.mean(x) for x in zip(*fraction_final, strict=True)]

            colors = [f"C{ii}" for ii in range(len(coverages))]
            colors[-1] = "none"
            # Top left plot (Row 0, Column 0)
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.set_title("Coverage")
            ax1.pie(coverages, colors=colors)

            # Add outer circle to ax1
            circle1 = plt.Circle((0, 0), 1, facecolor="none", edgecolor="black", linewidth=1.5)
            ax1.add_patch(circle1)

            # Top right plot (Row 0, Column 1)
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.set_title("Frac. cov. area")
            ax2.pie(fraction_of_covered_area, colors=colors)

            # Add outer circle to ax2
            circle2 = plt.Circle((0, 0), 1, facecolor="none", edgecolor="black", linewidth=1.5)
            ax2.add_patch(circle2)

            # Bottom double-length plot (Row 1, spans both Columns 0 and 1)
            ax3 = fig.add_subplot(gs[:, 1])
            ax3.set_title("Gap size distribution")
            sns.violinplot(np.hstack(gapsize_dist), ax=ax3, color="0.8")
            svg_buffer = io.BytesIO()
            plt.savefig(svg_buffer, format="svg", bbox_inches="tight")
            plt.close(fig)  # Clear memory

            svg_data = svg_buffer.getvalue()
            self.svg_widget.load(svg_data)
            self.svg_widget.renderer().setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)


        except (ValueError, TypeError, ValidationError) as e:
            self.error(f"Error processing compiled batch metrics:\n{e}")
        finally:
            self.run_group.setEnabled(True)
            self.progress_bar.setVisible(False)
            # self.run_group.setText("Run Simulation")

    def _on_simulation_error(self, exception: Exception) -> None:
        """Fallback callback handling background core crashes safely."""
        self.error(f"Simulation engine error:\n{exception}")
        self.run_group.setEnabled(True)
        self.progress_bar.setVisible(False)
        # self.run_group.setText("Run Simulation")

    def export_results(self) -> None:
        """Export the simulation results to JSON, HDF5, Pickle, or zipped CSVs."""


    def error(self, msg: str) -> None:
        """Handle the errors.

        :param msg: Error message to display in a new window.
        """
        QMessageBox.critical(self, "Input Error", msg)


class MoleculeGeneration(QWidget):
    """Molecule layout configuration dashboard tab view.

    Handles dynamic generation of geometric molecule polygon shapes via reflective
    library lookups, updates parameters on the fly, and lists them inside a tracking layout.
    """

    def __init__(self, state: AppState) -> None:
        """Initialise settings storage engines and compile separate view columns.

        :param state: AppState instance for communication between tabs.
        """
        super().__init__()

        self.param_widgets: dict[str, QSpinBox | QDoubleSpinBox | FilePickerWidget | QLineEdit]  = {}
        """"Parameter widgets derived from molecule function signatures."""
        self.opt_checkboxes: dict[str, QCheckBox] = {}
        """"Optional checkbox widgets derived from molecule function signatures."""
        self._settings = QSettings(type(self).__name__)
        """Persistent platform configuration handle cached between user runtime sessions."""

        self.state = state
        """Shared application state cache container."""

        # Initialise data storage metrics
        self._init_data_storage()

        # Build the three core panel containers
        left_container = self._build_left_panel()
        scroll_area = self._build_center_panel()
        right_container = self._build_right_panel()

        # Assemble components into the splitter layout interface
        self._assemble_layout(left_container, scroll_area, right_container)

    def _init_data_storage(self) -> None:
        """Initialise internal state tracking arrays and counting iterations."""
        self.mol_list_counter: count[int] = count()
        """Thread-safe sequential index iterator generating unique molecule instance identifier tags."""

        self.mol_params_list: list[MoleculeParameters] = []
        """List of MoleculeParameters dataclasses."""

    def _build_left_panel(self) -> QWidget:
        """Construct the left parameters control dashboard and link active list triggers.

        :return: A populated structural container pane acting as the configuration panel.
        """
        container = QWidget()
        self.controls_layout = QVBoxLayout(container)
        """Layout frame coordinating selection toggles and configuration property fields."""

        self.add_molecule_button = QPushButton("Add new molecule")
        """Action button triggering instance generation from the current profile layout."""
        self.controls_layout.addWidget(self.add_molecule_button)

        self.func_dropdown = QComboBox()
        """Selection field populated with valid introspected molecule generator workflows."""
        self.controls_layout.addWidget(QLabel("Select molecule"), alignment=Qt.AlignmentFlag.AlignTop)
        self.controls_layout.addWidget(self.func_dropdown, alignment=Qt.AlignmentFlag.AlignTop)

        # Discover generators using introspective library lookups
        self.generators = self._discover_molecule_generators()
        """Registry cache linking user-facing label text keys directly to underlying library callables."""

        self.func_dropdown.addItems(self.generators.keys())
        self.func_dropdown.currentTextChanged.connect(self._update_func_dropdown)

        # Parameter group layout setup
        mol_param_group = QGroupBox("Parameters")
        mol_param_layout = QVBoxLayout(mol_param_group)
        mol_param_layout.addWidget(QLabel("Mouse over parameter for tooltip."), alignment=Qt.AlignmentFlag.AlignTop)

        self.param_widgets: dict[str, QLineEdit | QDoubleSpinBox | QSpinBox | QFileDialog]
        """Active reference tracking field maps mapping variable names to their raw UI input views."""

        self.opt_checkboxes: dict[str, QCheckBox]
        """Active state checkboxes controlling presence flags for optional properties or switches."""

        self.param_layout = QVBoxLayout()
        """Parameter layout."""
        mol_param_layout.addLayout(self.param_layout)

        self.controls_layout.addWidget(mol_param_group, alignment=Qt.AlignmentFlag.AlignTop)

        target_index = self._fetch_setting("current_molecule", 0)

        # 4. Handle the index 0 edge case manually if it matches the default initial index
        if target_index == 0:
            # Force execution since setCurrentIndex(0) won't trigger a change event
            self.build_param_inputs(self.func_dropdown.currentText())
            self.plot_molecule()
        else:
            self.func_dropdown.setCurrentIndex(target_index)

        self.output_label = QLabel("")
        """Status indicator updating real-time compilation info or syntax exceptions."""
        self.controls_layout.addWidget(self.output_label)

        return container

    def _update_func_dropdown(self, name: str) -> None:
        """Update when func dropdown changes.

        :param name: The name of the dropdown option.
        """
        self.build_param_inputs(name)
        self.plot_molecule()

    @staticmethod
    def _discover_molecule_generators() -> dict[str, Callable[[P_mol.kwargs], Polygon]]:
        """Isolate reflection logic filtering usable library structural definitions.

        :return: A sorted lookup dict mapping valid function names to execution references.
        """
        temp_generators: dict[str, Callable[[P_mol.kwargs], Polygon]] = {
            name: func
            for name, func in molecule_lib.__dict__.items()
            if inspect.isfunction(func)
            and not name.startswith("_")
            and func.__module__ == molecule_lib.__name__
            and inspect.signature(func).return_annotation in {"Polygon", "dict[str, str | float | list[str] | None]"}
        }
        return dict(sorted(temp_generators.items()))

    def _build_center_panel(self) -> QScrollArea:
        """Construct the viewport frame area housing the centered vector graphics.

        :return: A scroll area wrapper managing the interactive central viewport.
        """
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        scroll_container = QWidget()
        container_layout = QGridLayout(scroll_container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        self.svg_widget = ZoomableSvgWidget()
        """Custom structural viewport rendering vector polygon outlines."""
        self.svg_widget.setMinimumSize(600, 600)
        self.svg_widget.renderer().setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)

        container_layout.addWidget(self.svg_widget, 0, 0, Qt.AlignmentFlag.AlignCenter)
        scroll_area.setWidget(scroll_container)

        return scroll_area

    def _build_right_panel(self) -> QWidget:
        """Construct the right tracking grid columns managing existing records.

        :return: A secondary control panel listing items added to the current system context.
        """
        container = QWidget()
        right_col = QVBoxLayout(container)

        group = QGroupBox("Molecules")
        group_layout = QVBoxLayout(group)

        self.molecule_list_widget = ReorderableListWidget()
        """List widget selection tool indicating current added molecule configurations."""
        self.molecule_list_widget.currentItemChanged.connect(self.show_molecule_settings)
        self.molecule_list_widget.itemsMoved.connect(self.sync_list_order)
        group_layout.addWidget(self.molecule_list_widget)

        self.delete_btn = QPushButton("Delete Selected Molecule")
        """Action button triggering array removal operations from local caches."""
        self.delete_btn.clicked.connect(self.delete_molecule)
        group_layout.addWidget(self.delete_btn)

        right_col.addWidget(group)

        return container

    def _assemble_layout(self, left: QWidget, center: QScrollArea, right: QWidget) -> None:
        """Unify sub-panels inside the scalable horizontal splitter framework.

        :param left: Parameter selection pane widget.
        :param center: Scroll pane holding vector outputs.
        :param right: Management column listing generated arrays.
        """
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        """Scalable divider framework managing responsive interface margins."""

        self.main_splitter.addWidget(left)
        self.main_splitter.addWidget(center)
        self.main_splitter.addWidget(right)

        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 3)
        self.main_splitter.setStretchFactor(2, 1)

        root_layout = QVBoxLayout(self)
        root_layout.addWidget(self.main_splitter)

    def _fetch_setting(self, name: str, default: T_co, return_type: type[T_co] | None = None) -> T_co:
        """Fetch settings by checking if they exist followed by their value.

        :param name: The name of the setting to fetch.
        :param default: The default value to return if the setting does not exist.
        :param return_type: The default return type if the setting exists. If not given, type(default) is used.
        :returns: The setting value if it exists, or else the default.
        """
        check_type: type[T_co] = type(default) if return_type is None else return_type
        return cast("T_co", self._settings.value(name, defaultValue=default, type=check_type))

    def _delete_previous_layout(self) -> None:
        """Recursively delete the layout of the previous molecule parameters."""
        def clear_layout(layout: QVBoxLayout | QHBoxLayout | QGridLayout | None) -> None:
            """Clear the layout by deleting its widgets or traversing its child layouts.

            :param layout: The layout to clear.
            """
            if layout is None:
                return

            while layout.count():
                item = layout.takeAt(0)

                # Use structural pattern matching to safely handle the item type
                match item.widget(), item.layout():
                    case (widget, _) if widget is not None:
                        # It is a widget container item
                        widget.deleteLater()

                    case (_, child_layout) if child_layout is not None:
                        # It is a nested layout item; recurse down first, then delete it
                        clear_layout(child_layout)
                        child_layout.deleteLater()

                    case _:
                        # It is a spacer item or an empty container
                        del item

        clear_layout(self.param_layout)

    def build_param_inputs(self, func_name: str) -> None:
        """Inspect a generator function signature to build a parameter layout frame.

        Clears existing child controls, parses required types from function type annotations,
        configures dynamic tooltip data, and maps live text update signalling pipelines.

        :param func_name: Target library function name registry string.
        """
        self._delete_previous_layout()
        self._settings.setValue("current_molecule", self.func_dropdown.currentIndex())

        func = self.generators[func_name]
        sig = inspect.signature(func)
        param_docs = extract_param_docs(func)

        if func_name == "first_time_loader":
            launch_loader_button = QPushButton("Launch first time loader")
            launch_loader_button.setToolTip("Start the first_time_loader script in a separate window.")
            launch_loader_button.clicked.connect(self.launch_first_time_loader)
            self.param_layout.addWidget(launch_loader_button)

        self.param_widgets = {}
        self.opt_checkboxes = {}

        param_grid = QGridLayout()
        for idx, (name, param) in enumerate(sig.parameters.items()):
            default = param.default
            is_optional: bool = default is None
            is_required: bool = param.default is inspect.Parameter.empty
            row = QHBoxLayout()

            name_label = QLabel(name.replace("_", " "))
            if name in param_docs:
                name_label.setToolTip(param_docs[name])
            param_grid.addWidget(name_label, idx, 0)

            # Route type dispatching to dedicated factory helper method
            widget = self._create_param_widget(param.annotation, default)

            if not (is_required or is_optional):
                match widget:
                    case QSpinBox() | QDoubleSpinBox():
                        widget.setValue(default)
                    case QLineEdit():
                        widget.setText(str(default))

            if is_optional:
                checkbox = QCheckBox()
                checkbox.setChecked(False)
                checkbox.setToolTip("If unchecked, the value is None")
                self.opt_checkboxes[name] = checkbox
                row.addWidget(checkbox, alignment=Qt.AlignmentFlag.AlignVCenter)

                widget.setEnabled(False)
                checkbox.toggled.connect(widget.setEnabled)

            if name in param_docs:
                widget.setToolTip(param_docs[name])

            row.addWidget(widget, alignment=Qt.AlignmentFlag.AlignVCenter)
            self.param_widgets[name] = widget
            param_grid.addLayout(row, idx, 1)

        self.param_layout.addLayout(param_grid)
        self.param_layout.addWidget(_make_horizontal_line())

        # Delegate secondary form components to helpers
        self._build_symmetry_controls()
        self.param_layout.addWidget(_make_horizontal_line())
        self._build_action_buttons()

    @staticmethod
    def _create_param_widget(annotation: str, default: str | float | inspect.Parameter.empty) -> QSpinBox | QDoubleSpinBox | FilePickerWidget | QLineEdit:
        """Create param widget using factory strategy translating library type hints to matching user input views.

        :param annotation: The raw string signature representation of the type hint.
        :param default: The underlying fallback data default value assigned to the flag.
        :return: A customised interactive input container widget subclass.
        :raises TypeError: If an unmapped or exotic data structure type is processed.
        """
        default_max: int = 999
        widget:  QSpinBox | QDoubleSpinBox | FilePickerWidget | QLineEdit
        match annotation:
            case "float" | "PositiveFloat" | "NonNegativeFloat" | "float | None":
                widget = QDoubleSpinBox()
                min_float_val: float = -999.0
                if annotation == "PositiveFloat":
                    min_float_val = 0.0001
                    if not isinstance(default, inspect.Parameter.empty):
                        widget.setValue(1.0)
                elif annotation == "NonNegativeFloat":
                    min_float_val = 0.0
                widget.setRange(min_float_val, default_max)
                widget.setDecimals(4)
                widget.setSingleStep(0.1)
                return widget

            case "int" | "PositiveInt":
                min_int_val: int = 1 if annotation == "PositiveInt" else -999
                widget = QSpinBox()
                widget.setRange(min_int_val, default_max)
                return widget

            case "FilePath":
                return FilePickerWidget()

            case "str | list[str] | None":
                return QLineEdit()

            case _:
                errmsg = f"Unsupported parameter annotation: '{annotation}'."
                raise TypeError(errmsg)

    def sync_list_order(self, old_index: int, new_index: int) -> None:
        """Take the row transformation from list A and applies it programmatically to list B.

        The ReorderableListWidget allows for items to be drag/dropped. This function links the reordering.

        :param old_index: The original index of the item changing position.
        :param new_index: The new index to which the item is moved.
        """
        # Pop the item out of its old position in List B
        taken_item = self.mol_params_list.pop(old_index)

        # Insert it into the exact same new position
        if taken_item:
            self.mol_params_list.insert(new_index, taken_item)
            self.state.molecule_param_list = self.mol_params_list

    def _build_symmetry_controls(self) -> None:
        """Assemble the geometric shape matrix transformation property grid layouts."""
        self.refl_sym: bool = False
        """Default value of reflection symmetry."""
        self.rot_sym: int = 1
        """Default value of rotation symmetry."""
        self.rot_cnt: int = 360
        """Default value of rotation count."""

        symmetry_options_layout = QGridLayout()
        refl_sym_label = QLabel("Reflection symmetry")
        refl_sym_tooltip_text = "Set checked if the molecule has reflection symmetry (symmetric group Cn → Dn)."
        refl_sym_label.setToolTip(refl_sym_tooltip_text)

        self.refl_sym_checkbox = QCheckBox()
        """Reflection symmetry checkbox, corresponding to True (checked) or False (unchecked)."""
        self.refl_sym_checkbox.setChecked(self.refl_sym)
        self.refl_sym_checkbox.setToolTip(refl_sym_tooltip_text)

        self.rot_sym_label = QLabel("Rotation symmetry")
        """Rotation symmetry label."""

        self.rot_sym_spinbox = QSpinBox()
        """Rotation symmetry spinbox, for non-negative integers."""
        self.rot_sym_spinbox.setMinimum(0)
        self.rot_sym_spinbox.setValue(self.rot_sym)
        self._update_symmetry_tooltip(self.refl_sym)
        self.refl_sym_checkbox.toggled.connect(self._update_symmetry_tooltip)

        rot_cnt_label = QLabel("Rotation count")
        rot_cnt_tooltip_text = "Number of rotations to be used for the molecule. The step size is 360/n°."
        rot_cnt_label.setToolTip(rot_cnt_tooltip_text)

        self.rot_cnt_spinbox = QSpinBox()
        """Rotation count spinbox, for positive (non-zero) integers."""
        self.rot_cnt_spinbox.setMinimum(1)
        self.rot_cnt_spinbox.setMaximum(99999)
        self.rot_cnt_spinbox.setValue(self.rot_cnt)
        self.rot_cnt_spinbox.setToolTip(rot_cnt_tooltip_text)

        symmetry_options_layout.addWidget(refl_sym_label, 0, 0)
        symmetry_options_layout.addWidget(self.refl_sym_checkbox, 0, 1)
        symmetry_options_layout.addWidget(self.rot_sym_label, 1, 0)
        symmetry_options_layout.addWidget(self.rot_sym_spinbox, 1, 1)
        symmetry_options_layout.addWidget(rot_cnt_label, 2, 0)
        symmetry_options_layout.addWidget(self.rot_cnt_spinbox, 2, 1)

        self.param_layout.addLayout(symmetry_options_layout)

    def _build_action_buttons(self) -> None:
        """Map active interactive preview checkboxes and form processing buttons."""
        molecule_buttons = QHBoxLayout()
        self.show_molecule_checkbox = QCheckBox("Plot Molecule")
        """Checkbox whether to show the molecule."""
        self.show_molecule_checkbox.setToolTip("Plot the molecule")
        # self.show_molecule_checkbox.setChecked(self.func_dropdown.currentText != "first_time_loader")
        self.show_molecule_checkbox.toggled.connect(self.plot_molecule)
        molecule_buttons.addWidget(self.show_molecule_checkbox)

        widget_object: QLineEdit | QDoubleSpinBox | QSpinBox | FilePickerWidget
        for widget_object in self.param_widgets.values():
            if not isinstance(widget_object, QLineEdit | QDoubleSpinBox | QSpinBox):
                continue
            if isinstance(widget_object, QLineEdit):
                widget_object.textChanged.connect(self.plot_molecule)
            else:
                widget_object.valueChanged.connect(self.plot_molecule)

        for optional_checkbox in self.opt_checkboxes.values():
            optional_checkbox.toggled.connect(self.plot_molecule)

        add_molecule_button = QPushButton("Add Molecule")
        add_molecule_button.clicked.connect(self.add_molecule)
        add_molecule_button.setToolTip("Add molecule to list of molecules in simulation")
        molecule_buttons.addWidget(add_molecule_button)

        self.param_layout.addLayout(molecule_buttons)

    def _update_symmetry_tooltip(self, is_checked: bool) -> None:
        """Update the tooltip of the rotation symmetry label and spinbox.

        :param is_checked: True if there is reflection symmetry, False otherwise.
        """
        symmetry_group = "D" if is_checked else "C"
        circle_group = "O(2)" if is_checked else "SO(2)"

        rot_sym_tooltip_text = textwrap.dedent(f"""\
            Keep 1 for no rotation symmetry (symmetric group {symmetry_group}1),
            2 for 180° symmetry ({symmetry_group}2),
            3 for 120° symmetry ({symmetry_group}3),
            n for 360/n° symmetry ({symmetry_group}n),
            Special case: 0 for circle symmetry ({circle_group}).""",
        )

        for widget in [self.rot_sym_label, self.rot_sym_spinbox]:
            widget.setToolTip(rot_sym_tooltip_text)

    def launch_first_time_loader(self) -> None:
        """Launch the first time loader from molecule_lib.

        If no file path has been provided, prompt the user to add one before running the first time loader.
        """
        if not self.param_widgets["file_name"].text():
            self.param_widgets["file_name"].browse_button.click()
        output = molecule_lib.first_time_loader(self.param_widgets["file_name"].text())
        for first_time_key, first_time_value in output.items():
            if first_time_value is not None:
                if isinstance(self.param_widgets[first_time_key], QLineEdit | FilePickerWidget):
                    fill_str: str = ",".join(first_time_value) if isinstance(first_time_value, list) else first_time_value
                    self.param_widgets[first_time_key].setText(fill_str)
                else:
                    self.param_widgets[first_time_key].setValue(first_time_value)
                if first_time_key in self.opt_checkboxes:
                    self.opt_checkboxes[first_time_key].setChecked(True)

    def get_param_values(self) -> dict[str, float | int | str | list[str] | None]:
        """Extract current user inputs from widgets back into a data dictionary.

        :return: Dictionary containing the key-value pairs of the parameters.
        """
        values: dict[str, float | int | str | list[str] | None] = {}
        name: str
        widget: QWidget

        for name, widget in self.param_widgets.items():
            # If the widget is disabled, the optional checkbox was unchecked -> value is None
            if not widget.isEnabled():
                continue

            # Extract value based on the PySide6/PyQt6 widget type
            match widget:
                case QSpinBox() | QDoubleSpinBox():
                    values[name] = widget.value()

                case QLineEdit():
                    values[name] = widget.text()

                case FilePickerWidget():
                    values[name] = widget.text()

                case _:
                    values[name] = None

        return values

    def error(self, msg: str) -> None:
        """Handle the errors.

        Please open a ticket if this happens when it should not.

        :param msg: Error message.
        """
        QMessageBox.critical(self, "Input Error", msg)

    def plot_molecule(self) -> None:
        """Plot the molecule."""
        if not self.show_molecule_checkbox.isChecked():
            return
        molecule_func = self.generators[self.func_dropdown.currentText()]
        molecule_dict = self.get_param_values()
        if molecule_func.__name__ == "first_time_loader":
            molecule_func = molecule_lib.xyz_reader
        try:
            svg_io = io.BytesIO()
            molecule_lib.save_molecule_svg(molecule_func(**molecule_dict), filename=svg_io)
            svg_data = svg_io.getvalue()
            self.svg_widget.load(svg_data)
        except ValueError as e:
            self.error(str(e))

    def add_molecule(self) -> None:
        """Add a molecule to the list of molecules to use."""
        current_func_name =  self.func_dropdown.currentText()
        current_func_name = current_func_name if current_func_name != "first_time_loader" else "xyz_reader"
        molecule_func = self.generators[current_func_name]
        molecule_dict = self.get_param_values()
        # if molecule_func.__name__ == "first_time_loader":
        #     molecule_func = molecule_lib.xyz_reader

        try:
            result = molecule_func(**molecule_dict)
        except ValidationError as e:
            self.error(str(e))
            return

        name = self.func_dropdown.currentText() if "file_name" not in molecule_dict else Path(molecule_dict["file_name"]).name

        # Update dropdown
        index = next(self.mol_list_counter)
        label = f"{name} #{index}"
        self.molecule_list_widget.addItem(label)

        mol_params = MoleculeParameters(
            index=index,
            function_name=current_func_name,
            label=label,
            polygon=result,
            settings=molecule_dict,
            refl_sym=self.refl_sym_checkbox.isChecked(),
            rot_sym=self.rot_sym_spinbox.value(),
            rot_cnt=self.rot_cnt_spinbox.value(),
        )

        self.mol_params_list.append(mol_params)
        self.state.molecule_param_list = self.mol_params_list

        self.output_label.setText(f"Added: {name}")

    def delete_molecule(self) -> None:
        """Delete the current selected molecule."""
        # Hitting the delete button without a selection results in -1.
        idx = self.molecule_list_widget.currentRow()
        if idx < 0:
            return

        del self.mol_params_list[idx]
        self.molecule_list_widget.takeItem(idx)
        self.state.molecule_param_list = self.mol_params_list

        self.output_label.setText("Molecule deleted")
        self.molecule_list_widget.clearSelection()
        self.molecule_list_widget.setCurrentItem(QListWidgetItem(None))

    def show_molecule_settings(self) -> None:
        """Show the settings of this molecule."""
        current_idx: int = self.molecule_list_widget.currentRow()
        if current_idx < 0 or current_idx >= len(self.mol_params_list):
            return
        current_name: str = self.mol_params_list[current_idx].function_name
        match_idx: int = self.func_dropdown.findText(current_name, Qt.MatchFlag.MatchExactly)
        self.func_dropdown.setCurrentIndex(match_idx)
        for key, val in self.mol_params_list[current_idx].settings.items():
            if val is None:
                continue
            current_param: QSpinBox | QDoubleSpinBox | QLineEdit | FilePickerWidget = self.param_widgets[key]
            if isinstance(current_param, QSpinBox | QDoubleSpinBox) and isinstance(val, int | float):
                current_param.setValue(val)
            elif isinstance(current_param, QLineEdit | FilePickerWidget) and isinstance(val, str):
                current_param.setText(val)
            else:
                errmsg: str = f"Type mismatch: Param type {type(current_param)} is not compatible with val {type(val)}"
                raise TypeError(errmsg)

            if key in self.opt_checkboxes:
                self.opt_checkboxes[key].setChecked(True)
        self.show_molecule_checkbox.setChecked(True)


class FilePickerWidget(QWidget):
    """Widget to help pick a file.

    :cvar text: Property parameter 'text', with getter and setter functions.
    """

    def __init__(self, parent: QWidget | None = None, placeholder: str = "Select a file...") -> None:
        """Initialise the file-picker widget.

        :param parent: Parent widget.
        :param placeholder: Placeholder text to display in the selector box.
        """
        super().__init__(parent)
        self._settings = QSettings(type(self).__name__)
        """Load the settings between sessions."""

        # Layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.line_edit = QLineEdit()
        """Line edit to show the path and hold the actual value."""
        self.line_edit.setPlaceholderText(placeholder)

        # Browse button
        self.browse_button = QPushButton("")
        """Button to browse files."""
        self.browse_button.setIcon(QIcon.fromTheme(QIcon.ThemeIcon.FolderOpen))
        self.browse_button.clicked.connect(self.open_file_dialog)

        # Add to layout
        layout.addWidget(self.line_edit)
        layout.addWidget(self.browse_button)

    def _get_text(self) -> str:
        """Get the current file path text.

        :returns: The current file path text string.
        """
        return self.line_edit.text()

    @Slot(str)
    def setText(self, value: str) -> None:  # noqa: N802
        """Set the file path text.

        :param value: The new file path text string.
        """
        self.line_edit.setText(value)

    text = Property(str, fget=_get_text, fset=setText, user=True)

    def _fetch_setting(self, name: str, default: T_co, return_type: type[T_co] | None = None) -> T_co:
        """Fetch settings by checking if they exist followed by their value.

        :param name: The name of the setting to fetch.
        :param default: The default value to return if the setting does not exist.
        :param return_type: The default return type if the setting exists. If not given, type(default) is used.
        :returns: The setting value if it exists, or else the default.
        """
        check_type = type(default) if return_type is None else return_type
        return cast("T_co", self._settings.value(name, defaultValue=default, type=check_type))

    def open_file_dialog(self) -> None:
        """Dialogue to display when selecting a file."""
        # Native OS file dialogue
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            caption="Select File",
            dir=self._fetch_setting("last_visited_directory", default=""),  # Start at current directory
            filter="XYZ File (*.xyz)",
        )
        if file_path:
            self.line_edit.setText(file_path)
            self._settings.setValue("last_visited_directory", str(Path(file_path).parent))

    def text(self) -> str:
        """Get the text of the box being edited.

        :return: Text of the box being edited.
        """
        return self.line_edit.text()


class SurfaceGeneration(QWidget):
    """Surface generation dashboard tab view.

    Provides control inputs for generating geometric lattice surfaces and
    displays the resulting surface within an interactive, centered viewer.
    """

    def __init__(self, state: AppState) -> None:
        """Initialise user widgets and assemble geometric layout wrappers.

        :param state: AppState object to share information between tabs.
        """
        super().__init__()
        self.state = state
        """Shared application state cache container."""

        self.surface_count: int = 50
        """Default surface site count."""

        self.real_surface_count: int = 50
        """Default computed surface site count."""

        self.stored_params: SurfaceParameters
        """Parameters of the surface, to be communicated between tabs."""

        self._init_validators()

        # Initialise panels
        left_container = self._build_left_panel()
        scroll_area = self._init_svg_view()

        # Assemble components directly inside the splitter framework
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        """Main splitter of the window."""
        self.main_splitter.addWidget(left_container)
        self.main_splitter.addWidget(scroll_area)

        # Configure layout stretching rules (1 unit left panel, 3 units centre viewport)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 3)

        # Clean up root assignment
        root_layout = QHBoxLayout(self)
        root_layout.addWidget(self.main_splitter)

    def _init_validators(self) -> None:
        """Instantiate validation models for text constraint processing."""
        self._gt_one_validator = QIntValidator(bottom=1)
        self._pos_float_validator = QDoubleValidator(bottom=0.0)

    def _build_left_panel(self) -> QWidget:
        """Construct the left controls container pane layout.

        :return: A populated structural layout container.
        """
        container = QWidget()
        layout = QVBoxLayout(container)

        # Surface configuration selector.
        layout.addWidget(QLabel("Surface Type:"), alignment=Qt.AlignmentFlag.AlignTop)
        self.surface_dropdown = QComboBox()
        """Selection box for geometry presets."""
        self.surface_dropdown.addItems(sorted(["hexagonal", "square", "honeycomb"]))
        self.surface_dropdown.setToolTip("Select surface type.")
        layout.addWidget(self.surface_dropdown, alignment=Qt.AlignmentFlag.AlignTop)

        # Theoretical count constraints input.
        layout.addWidget(QLabel("Optional surface site count (positive int):"), alignment=Qt.AlignmentFlag.AlignTop)
        self.site_count_input = QLineEdit()
        """Numeric entry field for requested surface site count."""
        self.site_count_input.setValidator(self._gt_one_validator)
        self.site_count_input.setPlaceholderText("e.g. 42")
        layout.addWidget(self.site_count_input, alignment=Qt.AlignmentFlag.AlignTop)

        # Evaluated real layout node calculation trackers.
        layout.addWidget(QLabel("Real surface site count:"), alignment=Qt.AlignmentFlag.AlignTop)
        self.real_site_count = QLabel()
        """Text label reflecting processed actual surface site count."""
        self.real_site_count.setText("50")
        self.real_site_count.setToolTip("The actual site count computed from the input count and the surface type.")
        layout.addWidget(self.real_site_count, alignment=Qt.AlignmentFlag.AlignTop)

        # Physical spacing distance parameters.
        layout.addWidget(QLabel("Lattice Spacing (optional, > 0 float):"), alignment=Qt.AlignmentFlag.AlignTop)
        self.lattice_input = QDoubleSpinBox()
        """Numeric entry field for lattice spacing."""
        self.lattice_input.setMinimum(0.0)
        self.lattice_input.setValue(1.0)
        self.lattice_input.setDecimals(2)
        self.lattice_input.setSingleStep(0.01)
        self.lattice_input.setAccelerated(True)
        self.lattice_input.setSuffix(" Å")
        self.lattice_input.setToolTip("Numeric entry field for lattice spacing.")
        layout.addWidget(self.lattice_input, alignment=Qt.AlignmentFlag.AlignTop)

        # Control trigger processing elements.
        self.generate_surface_button = QPushButton("Generate Surface")
        """Trigger execution pipeline for layout generation code."""
        self.generate_surface_button.setToolTip("Plot and store the surface.")
        layout.addWidget(self.generate_surface_button, alignment=Qt.AlignmentFlag.AlignTop)

        # Establish signalling loops.
        self.site_count_input.textChanged.connect(self._get_real_surface_site_count)
        self.surface_dropdown.currentIndexChanged.connect(self._get_real_surface_site_count)
        self.generate_surface_button.clicked.connect(self.generate_surface)

        # Force components to stick tight to the top boundary layout.
        layout.addStretch()
        return container

    def _init_svg_view(self) -> QScrollArea:
        """Construct the graphics frame and isolate structural canvas centering.

        :return: A scroll container managing the viewport window frame.
        """
        self.svg_widget = ZoomableSvgWidget()
        """Custom render context displaying loaded vector data."""
        self.svg_widget.renderer().setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)

        # Create a container widget with a Grid Layout to centre the SVG
        container = QWidget()
        container_layout = QGridLayout(container)
        container_layout.addWidget(self.svg_widget, 0, 0, Qt.AlignmentFlag.AlignCenter)

        self.scroll_area = QScrollArea()
        """Interactive bounding box containing the centered layout viewport."""
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(container)

        return self.scroll_area

    def _get_real_surface_site_count(self) -> None:
        """Get the real surface site count."""
        default_count: int = 50
        surface_type: str = self.surface_dropdown.currentText()
        temp_count: str = self.site_count_input.text().strip()
        self.surface_count = default_count if not temp_count else int(temp_count)
        self.real_surface_count = self.surface_count
        if surface_type == "hexagonal":
            self.real_surface_count *= 2
        elif surface_type == "honeycomb":
            self.real_surface_count *= 4
        self.real_site_count.setText(str(self.real_surface_count))

    def generate_surface(self) -> None:
        """Generate an example surface."""
        seed_text = self.state.seed_input.text().strip()
        seed: int | None = None
        if seed_text:
            if not seed_text.isnumeric() or int(seed_text) < 0:
                self.error("Seed must be a positive integer")
                return
            seed = int(seed_text)

        # Validate lattice spacing
        lattice_text = self.lattice_input.value()
        lattice: float | None = None
        if lattice_text:
            try:
                lattice = float(lattice_text)
            except ValueError as e:
                self.error(str(e))
                return

        lattice_type = self.surface_dropdown.currentText()
        dark_mode_bool = QGuiApplication.instance().styleHints().colorScheme() == Qt.ColorScheme.Dark

        svg_buffer = io.BytesIO()

        surf_params:  dict[str, float | None | str | int] = {
            "lattice_a": lattice,
            "lattice_type": lattice_type,
            "seed": seed,
            "site_count": self.surface_count,
        }

        show_surface(
            **surf_params,
            filepath=svg_buffer,
            svg_flag=True,
            dark_mode_bool=dark_mode_bool,
        )
        svg_data = svg_buffer.getvalue()

        self.svg_widget.load(svg_data)
        self.svg_widget.renderer().setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)

        self.stored_params = SurfaceParameters(**surf_params)
        self.state.surface_params = self.stored_params

    def error(self, msg: str) -> None:
        """Handle the errors.

        :param msg: Error message to display in a new window.
        """
        QMessageBox.critical(self, "Input Error", msg)


class BackgroundTaskSignals(QObject):
    """Signals for the generic background worker.

    cvar finished: Emits the raw output data package.
    cvar progress: Emits the current simulation progress as a percentage integer.
    """

    finished = Signal(object)  # Emits the raw output data package
    error = Signal(Exception)  # Emits any exception caught during execution
    progress = Signal(int)  # Emits the integer percentage (0 to 100)

class BackgroundTask(QRunnable):
    """Executes a single blocking function call in the background thread pool."""

    def __init__(self, func: Callable[..., Any], *args: P.args, **kwargs: P.kwargs) -> None:
        """Initialise the BackgroundTask.

        :param func: Function to be executed.
        :param args: Positional arguments to be passed to the function.
        :param kwargs: Keyword arguments to be passed to the function.
        """
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.signals = BackgroundTaskSignals()

    @override
    def run(self) -> None:
        """Run the background task."""
        try:
            result = self.func(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except ValueError as e:
            self.signals.error.emit(e)


class ReorderableListWidget(QListWidget):
    """Reorderable list widget.

    :cvar itemsMoved: Custom signal that emits (old_row_idx, new_row_idx).
    """

    itemsMoved = Signal(int, int)  # noqa: N815

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise the ReorderableListWidget.

        :param parent: Parent widget.
        """
        super().__init__(parent)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)

    @override
    def dropEvent(self, event: QDropEvent) -> None:
        """Overridden drop event method.

        Define a new drop event that emits the old and the new position.

        :param event: Event object.
        """
        # Find the item currently selected (the one being dragged)
        moving_item = self.currentItem()
        if not moving_item:
            super().dropEvent(event)
            return

        old_row = self.row(moving_item)

        # QListWidget handles visual reordering
        super().dropEvent(event)

        # Find the item's new position after the move completes
        new_row = self.row(moving_item)

        # If the position actually changed, emit the signal
        if old_row != new_row:
            self.itemsMoved.emit(old_row, new_row)

def _make_horizontal_line() -> QFrame:
    """Create a horizontal line widget using a QFrame object.

    :returns: A horizontal line widget.
    """
    hline = QFrame()
    hline.setFrameShape(QFrame.Shape.HLine)
    hline.setFrameShadow(QFrame.Shadow.Sunken)
    return hline

def main() -> int:
    """Launch the adsorpy GUI.

    :returns: Return code.
    """
    app = QApplication(sys.argv)
    app.setOrganizationName("adsorpy")
    app.setApplicationName("adsorpy-app")
    gui = AdsorpyGUI()
    gui.resize(1600, 900)
    gui.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
