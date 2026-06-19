# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""GUI module of adsorpy."""  # TODO: Make a new repo for this!

from __future__ import annotations

import inspect
import io
import re
import sys
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, ParamSpec, Self, TypeVar, cast, override

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QRegularExpression, QSettings, Qt, Signal
from PySide6.QtGui import QAction, QDoubleValidator, QIcon, QIntValidator, QRegularExpressionValidator, QWheelEvent
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from adsorpy.molecule_lib import first_time_loader, xyz_reader
from src.adsorpy import molecule_lib
from src.adsorpy.run_simulation import run_simulation, show_surface

Tqob = TypeVar("Tqob", bound=QObject)
T = TypeVar("T", bound=bool | int | str | float)

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapely import Polygon

    P = ParamSpec("P", bound=int | float | str | list[str] | None)  # Helps with static type checkers.


def extract_param_docs(func: Callable) -> dict[str, str]:
    """Extract parameters and their types from the docstring of a function.

    This function is written for reStructuredText (rst) style docstrings.

    :param func: The function from which the docstring is extracted.
    :return: The dictionary of parameters and their types (as strings).
    :raises ValueError: If the function has no docstring.
    """
    doc = inspect.getdoc(func)
    if not doc:
        errmsg: str = f"Docstring of {func.__name__} is not defined."
        raise ValueError(errmsg)

    param_docs: dict[str, str] = {}
    lines = doc.splitlines()

    current_param: str | None = None
    buffer = []

    for line in lines:
        param_match = re.match(r"\s*:param\s+(\w+)\s*:\s*(.*)", line)
        if param_match:
            if current_param and buffer:
                param_docs[current_param] = " ".join(buffer).strip()

            current_param = param_match.group(1)
            buffer = [param_match.group(2).strip()]

        elif current_param and line.startswith("    "):
            buffer.append(line.strip())

    # Save last param
    if current_param and buffer:
        param_docs[current_param] = " ".join(buffer).strip()

    return param_docs


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
            self.setFixedSize(800, 600)

    @override
    def wheelEvent(self, event: QWheelEvent) -> None:
        """Override scroll wheel events.

        :param event: The QWheelEvent object.
        """
        modifiers = event.modifiers()
        scroll_area = self.window().findChild(QScrollArea)

        # 1. CTRL + SCROLL = ZOOM TO MOUSE CURSOR
        if modifiers == Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            scale = self.zoom_factor if delta > 0 else 1.0 / self.zoom_factor

            old_size = self.size()
            new_size = old_size * scale
            min_scale: int = 100
            max_scale: int = 5000

            if min_scale < new_size.width() < max_scale:
                mouse_pos = event.position()
                self.setFixedSize(new_size)

                if scroll_area:
                    h_bar = scroll_area.horizontalScrollBar()
                    v_bar = scroll_area.verticalScrollBar()

                    new_h_val = int(h_bar.value() + mouse_pos.x() * (scale - 1.0))
                    new_v_val = int(v_bar.value() + mouse_pos.y() * (scale - 1.0))

                    h_bar.setValue(new_h_val)
                    v_bar.setValue(new_v_val)
            event.accept()

        # 2. SHIFT + SCROLL = LEFT / RIGHT PANNING
        elif modifiers == Qt.KeyboardModifier.ShiftModifier:
            if scroll_area:
                h_bar = scroll_area.horizontalScrollBar()
                # Determine scroll distance based on wheel rotation (typically 120 units per notch)
                steps = event.angleDelta().y()
                # Invert steps so scrolling up goes left, scrolling down goes right
                h_bar.setValue(h_bar.value() - steps)
            event.accept()

        # 3. NO MODIFIERS = STANDARD VERTICAL PANNING
        else:
            event.ignore()  # Hands control over to QScrollArea naturally


class AutoStateMeta(type(QObject), Generic[Tqob]):
    """Metaclass for AppState to automatically communicate between tabs."""

    fields: ClassVar[dict[str, type]]

    def __new__(cls: type[Self], name: str, bases: tuple[type, ...], attrs: dict[str, Any]) -> type[Self]:
        """Create an AutoState class instance.

        :param name: The name of the class.
        :param bases: The base classes.
        :param attrs: The class attributes.
        :return: The AutoState class instance.
        """
        fields = attrs.get("fields", {})

        for field_name, field_type in fields.items():
            signal_name = f"{field_name}Changed"
            private_name = f"_{field_name}"

            attrs[signal_name] = Signal(field_type)

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

        return super().__new__(cls, name, bases, attrs)

    def __call__(cls, *args: P.args, **kwargs: P.kwargs) -> Tqob:
        """Instantiate the class and auto-initialise its fields.

        :param args: Positional arguments.
        :param kwargs: Keyword arguments.
        :returns: The initialised class instance.
        """
        obj = super().__call__(*args, **kwargs)

        # Auto-initialise private fields
        for field_name in cls.fields:
            private_name = f"_{field_name}"
            if not hasattr(obj, private_name):
                setattr(obj, private_name, None)

        return obj


class AppState(QObject, metaclass=AutoStateMeta):
    """AppState class to communicate between tabs."""

    filepath: str
    seed_input: QLineEdit
    count: int
    step_limit: int

    fields: ClassVar[dict[str, type]] = {
        "filepath": str,
        "seed_input": QLineEdit,
        "count": int,
        "step_limit": int,
    }


class MplCanvas(FigureCanvasQTAgg):
    """Plot canvas class."""

    def __init__(self) -> None:
        """Initialise plot canvas class."""
        self.fig = Figure(figsize=(16, 9))
        self.ax = self.fig.add_subplot(111)
        # self.plot_widget = pg.PlotWidget()
        super().__init__(self.fig)


class AdsorpyGUI(QMainWindow):
    """AdsorPy GUI."""

    def __init__(self) -> None:
        """Initialise the AdsorPy GUI."""
        super().__init__()

        self.setWindowTitle("Adsorpy Simulation GUI")

        menubar = self.menuBar()
        self.state = AppState()

        # File menu
        file_menu = menubar.addMenu("File")

        new_action = QAction(QIcon.fromTheme(QIcon.ThemeIcon.DocumentNew), "New Simulation", self)
        open_action = QAction(QIcon.fromTheme(QIcon.ThemeIcon.DocumentOpen), "Open…", self)
        save_action = QAction(QIcon.fromTheme(QIcon.ThemeIcon.DocumentSave), "Save", self)
        exit_action = QAction(QIcon("assets/door-open-out.png"), "Exit", self)
        exit_action.triggered.connect(self.close)

        file_menu.addAction(new_action)
        file_menu.addAction(open_action)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        # Documentation
        doc_action = QAction(QIcon("assets/book-open-list.png"), "Documentation (web)", self)
        doc_action.triggered.connect(
            lambda: webbrowser.open("https://joostfwmaas.github.io/AdsorPy/"),
        )

        # Wiki
        wiki_action = QAction(QIcon.fromTheme(QIcon.ThemeIcon.HelpFaq), "Wiki (web)", self)
        wiki_action.triggered.connect(
            lambda: webbrowser.open("https://github.com/JoostFWMaas/AdsorPy/wiki"),
        )

        # Report bug
        bug_action = QAction(QIcon("assets/bug--exclamation.png"), "Report bug (web)", self)
        bug_action.triggered.connect(
            lambda: webbrowser.open("https://github.com/JoostFWMaas/AdsorPy/issues"),
        )

        help_menu.addAction(doc_action)
        help_menu.addAction(wiki_action)
        help_menu.addAction(bug_action)

        central = SurfaceGeneration(self.state)
        self.setCentralWidget(central)

        main_layout = QHBoxLayout()
        central.setLayout(main_layout)

        controls_layout = QVBoxLayout()
        main_layout.addLayout(controls_layout)

        tabs = QTabWidget()
        tabs.addTab(GeneralSettings(self.state), "General")
        tabs.addTab(SurfaceGeneration(self.state), "Surface")
        tabs.addTab(MoleculeGeneration(self.state), "Molecule(s)")
        self.setCentralWidget(tabs)


class GeneralSettings(QWidget):
    """General settings generation widget."""

    def __init__(self, state: AppState) -> None:
        """Initialise the general settings widget.

        :param state: AppState shared state between tabs.
        """
        super().__init__()
        self.state = state

        main_layout = QHBoxLayout()
        controls_layout = QVBoxLayout()
        gt_one_validator = QIntValidator(bottom=1)
        # pos_float_validator = QDoubleValidator(bottom=0)
        seed_validator = QRegularExpressionValidator(regularExpression=QRegularExpression(r"^\d+$"))

        # Seed input
        controls_layout.addWidget(QLabel("Optional Seed (positive int):"))
        self.seed_input = QLineEdit()
        self.seed_input.setValidator(seed_validator)
        self.seed_input.setPlaceholderText("e.g. 23")  # Skidoo!
        controls_layout.addWidget(self.seed_input)
        self.state.seed_input = self.seed_input

        # Step limit input box
        controls_layout.addWidget(QLabel("Step limit (optional, > 0 int):"))
        self.step_limit = QLineEdit()
        self.step_limit.setPlaceholderText("e.g. 1")
        self.step_limit.setValidator(gt_one_validator)
        self.step_limit.textChanged.connect(self.on_step_limit_changed)
        controls_layout.addWidget(self.step_limit)

        main_layout.addLayout(controls_layout)

        self.canvas = MplCanvas()
        svg_buffer = io.BytesIO()
        self.canvas.fig.savefig(svg_buffer, format="svg", bbox_inches="tight")
        svg_data = svg_buffer.getvalue()
        self.svg_widget = QSvgWidget()
        self.svg_widget.renderer().setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        self.svg_widget.load(svg_data)

        # Add the SVG widget to layout
        main_layout.addWidget(self.svg_widget, stretch=1)

        self.setLayout(main_layout)

    def on_step_limit_changed(self, value: str) -> None:
        """Update the step limit value.

        :param value: The new step limit value.
        """
        self.state.step_limit = int(value) if value else -1


class MoleculeGeneration(QWidget):
    """Molecule generation widget."""

    def __init__(self, state: AppState) -> None:
        """Initialise the molecule generation widget.

        :param state: AppState shared state between tabs.
        """
        super().__init__()
        self._settings = QSettings("adsorpy", type(self).__name__)
        """Load the settings between sessions."""
        self.state = state

        self.main_layout = QHBoxLayout()
        self.controls_layout = QVBoxLayout()
        # gt_one_validator = QIntValidator(bottom=1)
        # pos_float_validator = QDoubleValidator(bottom=0)
        # seed_validator = QRegularExpressionValidator(regularExpression=QRegularExpression(r"^\d+$"))

        # Add molecule button
        self.add_molecule_button = QPushButton("Add new molecule")
        # self.add_molecule_button.clicked.connect(self.add_molecule)
        self.controls_layout.addWidget(self.add_molecule_button)
        self.main_layout.addLayout(self.controls_layout)

        self.func_dropdown = QComboBox()
        self.controls_layout.addWidget(QLabel("Select molecule"), alignment=Qt.AlignmentFlag.AlignTop)
        self.controls_layout.addWidget(self.func_dropdown, alignment=Qt.AlignmentFlag.AlignTop)

        temp_generators = {
            name: func
            for name, func in molecule_lib.__dict__.items()
            if inspect.isfunction(func)
            and not name.startswith("_")
            and func.__module__ == molecule_lib.__name__
            and inspect.signature(func).return_annotation in {"Polygon", "dict[str, str | float | list[str] | None]"}
        }

        self.generators: dict[str, Callable[[...], Polygon]] = dict(sorted(temp_generators.items()))

        self.func_dropdown.addItems(self.generators.keys())
        self.func_dropdown.currentTextChanged.connect(self.build_param_inputs)
        mol_param_group = QGroupBox("Parameters")
        mol_param_layout = QVBoxLayout()

        mol_param_layout.addWidget(QLabel("Mouse over parameter for tooltip."), alignment=Qt.AlignmentFlag.AlignTop)

        self.param_widgets: dict[str, QLineEdit | QDoubleSpinBox | QSpinBox | QFileDialog] = {}
        self.opt_checkboxes: dict[str, QCheckBox] = {}
        self.param_layout = QVBoxLayout()
        mol_param_layout.addLayout(self.param_layout)

        mol_param_group.setLayout(mol_param_layout)
        self.controls_layout.addWidget(mol_param_group, alignment=Qt.AlignmentFlag.AlignTop)
        self.func_dropdown.setCurrentIndex(self._fetch_setting("current_molecule", 0))

        # Molecule dropdown
        # controls_layout.addWidget(QLabel("Molecule:"))
        # self.molecule_dropdown = QComboBox()
        # self.molecule_names = list_public_molecules()
        # self.molecule_dropdown.addItems(self.molecule_names)
        # controls_layout.addWidget(self.molecule_dropdown)

        # Plot molecules
        # self.canvas = MplCanvas()
        # svg_buffer = io.BytesIO()
        # self.canvas.fig.savefig(svg_buffer, format="svg", bbox_inches="tight")
        # svg_data = svg_buffer.getvalue()
        self.svg_widget = ZoomableSvgWidget()
        self.svg_widget.renderer().setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        # self.svg_widget.renderer().setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        # self.svg_widget.load(svg_data)

        # Add the SVG widget to layout
        # layout.addWidget(svg_widget)
        # main_layout.addWidget(self.canvas, stretch=1)
        self.main_layout.addWidget(self.svg_widget, stretch=1)

        self.setLayout(self.main_layout)

        self.output_label = QLabel("")
        self.controls_layout.addWidget(self.output_label)
        self.main_layout.addLayout(self.controls_layout)

        right_col = QVBoxLayout()

        group = QGroupBox("Molecules")
        group_layout = QVBoxLayout()

        self.molecule_dropdown = QComboBox()
        group_layout.addWidget(self.molecule_dropdown)

        # Delete button
        self.delete_btn = QPushButton("Delete Selected Molecule")
        self.delete_btn.clicked.connect(self.delete_molecule)
        group_layout.addWidget(self.delete_btn)

        group.setLayout(group_layout)
        right_col.addWidget(group)

        # Add right column to main layout
        self.main_layout.addLayout(right_col)

        # Build initial parameter UI
        self.build_param_inputs(self.func_dropdown.currentText())

        # Internal molecule storage
        self.molecules = []

    def _fetch_setting(self, name: str, default: T, return_type: type[T] | None = None) -> T:
        """Fetch settings by checking if they exist followed by their value.

        :param name: The name of the setting to fetch.
        :param default: The default value to return if the setting does not exist.
        :param return_type: The default return type if the setting exists. If not given, type(default) is used.
        :returns: The setting value if it exists, or else the default.
        """
        check_type = type(default) if return_type is None else return_type
        return cast("T", self._settings.value(name, defaultValue=default, type=check_type))

    # ---------------------------------------------------------
    # Build parameter widgets dynamically
    # ---------------------------------------------------------
    def _delete_previous_layout(self) -> None:
        """Delete the layout of the previous molecule parameters."""
        while self.param_layout.count():
            item = self.param_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            # if it's a nested layout, clear it too
            child_layout = item.layout()
            if child_layout is not None:
                while child_layout.count():
                    child_item = child_layout.takeAt(0)
                    child_widget = child_item.widget()
                    if child_widget is not None:
                        child_widget.deleteLater()

    def build_param_inputs(self, func_name: str) -> None:
        """Build the parameter inputs.

        :param func_name: Name of the function
        """
        self._delete_previous_layout()
        self._settings.setValue("current_molecule", self.func_dropdown.currentIndex())

        func = self.generators[func_name]
        sig = inspect.signature(func)
        param_docs = extract_param_docs(func)
        default_max: int = 999

        if func_name == "first_time_loader":
            launch_loader_button = QPushButton("Launch first time loader")
            launch_loader_button.setToolTip("Start the first_time_loader script in a separate window.")
            launch_loader_button.clicked.connect(self.launch_first_time_loader)
            self.param_layout.addWidget(launch_loader_button)

        self.param_widgets = {}
        self.opt_checkboxes: dict[str, QCheckBox] = {}

        for name, param in sig.parameters.items():
            default = param.default
            is_optional: bool = default is None
            is_required: bool = param.default is inspect.Parameter.empty
            row = QHBoxLayout()

            name_label = QLabel(name)
            if name in param_docs:
                name_label.setToolTip(param_docs[name])
            row.addWidget(name_label, alignment=Qt.AlignmentFlag.AlignVCenter)

            # strip_params = param.annotation.split("|")
            # print(strip_params)

            match param.annotation:
                case "float" | "PositiveFloat" | "NonNegativeFloat" | "float | None":
                    widget = QDoubleSpinBox()
                    min_float_val: float = -999.0
                    if param.annotation == "PositiveFloat":
                        min_float_val = 0.0001
                        if not isinstance(default, inspect.Parameter.empty):
                            widget.setValue(1.0)
                    elif param.annotation == "NonNegativeFloat":
                        min_float_val = 0.0
                    widget.setRange(min_float_val, default_max)
                    widget.setDecimals(4)
                    widget.setSingleStep(0.1)

                case "int" | "PositiveInt":
                    min_int_val: int = 1 if param.annotation == "PositiveInt" else -999
                    widget = QSpinBox()
                    widget.setRange(min_int_val, default_max)

                case "FilePath":
                    widget = FilePickerWidget()

                case "str | list[str] | None":
                    widget = QLineEdit()

                case _:
                    errmsg: str = f"Unsupported parameter annotation: '{param.annotation}' for param '{name}'."
                    raise TypeError(errmsg)
                    # pass

            if not is_required and default is not None:
                match widget:
                    case QSpinBox() | QDoubleSpinBox():
                        widget.setValue(default)
                    case QLineEdit():
                        widget.setText(str(default))

            if is_optional:
                # Checkbox for optional parameters
                self.opt_checkboxes[name] = QCheckBox()
                self.opt_checkboxes[name].setChecked(False)
                row.addWidget(self.opt_checkboxes[name], alignment=Qt.AlignmentFlag.AlignVCenter)
                widget.setEnabled(False)
                self.opt_checkboxes[name].toggled.connect(widget.setEnabled)

            if name in param_docs:
                widget.setToolTip(param_docs[name])
            row.addWidget(widget, alignment=Qt.AlignmentFlag.AlignVCenter)
            self.param_widgets[name] = widget
            self.param_layout.addLayout(row)

        molecule_buttons = QHBoxLayout()
        self.show_molecule_checkbox = QCheckBox("Plot Molecule")
        self.show_molecule_checkbox.setToolTip("Plot the molecule")
        self.show_molecule_checkbox.setChecked(False)
        self.show_molecule_checkbox.toggled.connect(self.plot_molecule)
        molecule_buttons.addWidget(self.show_molecule_checkbox)
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

    def launch_first_time_loader(self) -> None:
        """Launch the first time loader from molecule_lib.

        If no file path has been provided, prompt the user to add one before running the first time loader.
        """
        if not self.param_widgets["file_name"].text():
            self.param_widgets["file_name"].browse_button.click()
        output = first_time_loader(self.param_widgets["file_name"].text())
        for first_time_key, first_time_value in output.items():
            if first_time_value is not None:
                if isinstance(self.param_widgets[first_time_key], QLineEdit | FilePickerWidget):
                    self.param_widgets[first_time_key].setText(first_time_value)
                else:
                    self.param_widgets[first_time_key].setValue(first_time_value)
                if first_time_key in self.opt_checkboxes:
                    self.opt_checkboxes[first_time_key].setChecked(True)

    def get_param_values(self) -> dict[str, float | int | str | list[str] | None]:
        """Extract current user inputs from widgets back into a data dictionary."""
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
        """
        QMessageBox.critical(self, "Input Error", msg)

    def plot_molecule(self) -> None:
        """Plot the molecule."""
        if not self.show_molecule_checkbox.isChecked():
            return
        molecule_func = self.generators[self.func_dropdown.currentText()]
        molecule_dict = self.get_param_values()
        if molecule_func.__name__ == "first_time_loader":
            molecule_func = xyz_reader
        try:
            svg_io = io.BytesIO()
            molecule_lib.save_molecule_svg(molecule_func(**molecule_dict), filename=svg_io)
            svg_data = svg_io.getvalue()
            self.svg_widget.load(svg_data)
        except ValueError as e:
            self.error(str(e))

    def add_molecule(self) -> None:
        """Add a molecule to the list of molecules to use."""
        func_name = self.func_dropdown.currentText()
        func = self.generators[func_name]

        kwargs = {}
        for name, (widget, checkbox) in self.param_widgets.items():
            if checkbox is not None and not checkbox.isChecked():
                # Optional parameter not enabled → skip it
                continue

            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                kwargs[name] = widget.value()
            else:
                text = widget.text()
                kwargs[name] = None if text == "" else text

        result = func(**kwargs)

        # Store molecule
        self.molecules.append(result)

        # Update dropdown
        label = f"{result['name']} #{len(self.molecules)}"
        self.molecule_dropdown.addItem(label)

        self.output_label.setText(f"Added: {label}")

    # ---------------------------------------------------------
    # Delete selected molecule
    # ---------------------------------------------------------
    def delete_molecule(self) -> None:
        """Delete the current selected molecule."""
        idx = self.molecule_dropdown.currentIndex()
        if idx < 0:
            return

        # Remove from internal list
        del self.molecules[idx]

        # Remove from dropdown
        self.molecule_dropdown.removeItem(idx)

        self.output_label.setText("Molecule deleted")


class FilePickerWidget(QWidget):
    """Widget to help pick a file."""

    def __init__(self, parent: QWidget | None = None, placeholder: str = "Select a file...") -> None:
        """Initialise the file-picker widget.

        :param parent: Parent widget.
        :param placeholder: Placeholder text to display in the selector box.
        """
        super().__init__(parent)
        self._settings = QSettings("adsorpy", type(self).__name__)
        """Load the settings between sessions."""

        # Layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Line edit to show the path and hold the actual value
        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText(placeholder)
        self.setText = self.line_edit.setText

        # Browse button
        self.browse_button = QPushButton("")
        self.browse_button.setIcon(QIcon.fromTheme(QIcon.ThemeIcon.FolderOpen))
        self.browse_button.clicked.connect(self.open_file_dialog)

        # Add to layout
        layout.addWidget(self.line_edit)
        layout.addWidget(self.browse_button)

    def _fetch_setting(self, name: str, default: T, return_type: type[T] | None = None) -> T:
        """Fetch settings by checking if they exist followed by their value.

        :param name: The name of the setting to fetch.
        :param default: The default value to return if the setting does not exist.
        :param return_type: The default return type if the setting exists. If not given, type(default) is used.
        :returns: The setting value if it exists, or else the default.
        """
        check_type = type(default) if return_type is None else return_type
        return cast("T", self._settings.value(name, defaultValue=default, type=check_type))

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
    """Surface generation widget."""

    def __init__(self, state: AppState) -> None:
        """Initialise surface generation widget.

        :param state: AppState shared state between tabs.
        """
        super().__init__()
        self.state = state
        # self.setWindowTitle("Surface Generation")

        main_layout = QHBoxLayout()
        controls_layout = QVBoxLayout()
        gt_one_validator = QIntValidator(bottom=1)
        pos_float_validator = QDoubleValidator(bottom=0)

        # Surface dropdown
        controls_layout.addWidget(QLabel("Surface Type:"))
        self.surface_dropdown = QComboBox()
        self.surface_dropdown.addItems(sorted(["hexagonal", "square", "honeycomb"]))
        controls_layout.addWidget(self.surface_dropdown)

        # Surface site count
        controls_layout.addWidget(QLabel("Optional surface site count (positive int):"))
        self.site_count_input = QLineEdit()
        self.site_count_input.setValidator(gt_one_validator)
        self.site_count_input.setPlaceholderText("e.g. 42")  # Answer to Ultimate Question of Life, Universe, Everything
        controls_layout.addWidget(self.site_count_input)

        # Real surface site count output
        controls_layout.addWidget(QLabel("Real surface site count:"))
        self.real_site_count = QLabel()
        self.real_site_count.setText("50")
        self.site_count_input.textChanged.connect(self._get_real_surface_site_count)
        self.surface_dropdown.currentIndexChanged.connect(self._get_real_surface_site_count)
        controls_layout.addWidget(self.real_site_count)

        # Lattice spacing input box
        controls_layout.addWidget(QLabel("Lattice Spacing (optional, > 0 float):"))
        self.lattice_input = QLineEdit()
        self.lattice_input.setValidator(pos_float_validator)
        self.lattice_input.setPlaceholderText("e.g. 1.0")
        controls_layout.addWidget(self.lattice_input)

        # Generate surface button
        self.generate_surface_button = QPushButton("Generate Surface")
        self.generate_surface_button.clicked.connect(self.generate_surface)
        controls_layout.addWidget(self.generate_surface_button)

        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)

        # Run button
        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.run_simulation)
        controls_layout.addWidget(self.run_button)

        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)

        # Plot canvas
        self.canvas = MplCanvas()
        svg_buffer = io.BytesIO()
        self.canvas.fig.savefig(svg_buffer, format="svg", bbox_inches="tight")
        svg_data = svg_buffer.getvalue()
        self.svg_widget = ZoomableSvgWidget()
        self.svg_widget.renderer().setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        self.svg_widget.load(svg_data)
        self.scroll_area = QScrollArea()

        # CRITICAL: This must be False so the inner widget can expand past the window size
        self.scroll_area.setWidgetResizable(False)

        # CRITICAL: Directly set the widget. Do NOT wrap it in an extra QWidget layout
        self.scroll_area.setWidget(self.svg_widget)

        # Add the SVG widget to layout
        # layout.addWidget(svg_widget)
        # main_layout.addWidget(self.canvas, stretch=1)
        main_layout.addWidget(self.scroll_area, stretch=1)

        self.setLayout(main_layout)

        self.surface_count: int = 50
        self.real_surface_count: int = 50

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
        lattice_text = self.lattice_input.text().strip()
        lattice: float | None = None
        if lattice_text:
            try:
                lattice = float(lattice_text)
            except ValueError as e:
                self.error(str(e))
                return

        lattice_type = self.surface_dropdown.currentText()

        # ------------------------------------------------------
        # Plot result
        # ------------------------------------------------------
        # self.canvas.ax.set_title("Adsorbed molecules")
        svg_buffer = io.BytesIO()
        self.canvas = show_surface(
            lattice_a=lattice,
            lattice_type=lattice_type,
            seed=seed,
            site_count=self.surface_count,
            filepath=svg_buffer,
            svg_flag=True,
        )
        svg_data = svg_buffer.getvalue()

        self.svg_widget.load(svg_data)
        self.svg_widget.renderer().setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        # Add the SVG widget to layout
        # self.canvas.addWidget(svg_widget)

    def run_simulation(self) -> None:
        """Run the simulation."""
        # Validate seed
        seed_text = self.state.seed_input.text().strip()
        seed: int | None = None
        if seed_text:
            if not seed_text.isnumeric() or int(seed_text) < 0:
                self.error("Seed must be a positive integer")
                return
            seed = int(seed_text)

        # Validate lattice spacing
        lattice_text = self.lattice_input.text().strip()
        lattice: float | None = None
        if lattice_text:
            try:
                lattice = float(lattice_text)
            except ValueError as e:
                self.error(str(e))
                return

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

        # Validate step limit
        step_limit_val: int | None = self.state.step_limit
        good_limit: bool = step_limit_val is not None and step_limit_val >= 0

        step_limit: int = int(step_limit_val) if good_limit else cast("int", get_run_sim_default("timestep_limit"))

        # molecule = self.molecule_dropdown.currentText()
        lattice_type = self.surface_dropdown.currentText()

        # Run simulation
        output = run_simulation(
            seed=seed,
            lattice_type=lattice_type,
            lattice_a=lattice,
            site_count=self.surface_count,
            timestep_limit=step_limit,
        )[-1]

        # ------------------------------------------------------
        # Plot result
        # ------------------------------------------------------
        # self.canvas.ax.clear()
        # plot_polygon(MultiPolygon(output.mol_data.stored_mirr_data["polygon"]), ax=self.canvas.ax, add_points=False)
        # self.canvas.ax.set_title("Adsorbed molecules")
        # self.canvas.ax.set_xlabel("x")
        # self.canvas.ax.set_ylabel("y")
        # self.canvas.ax.set_aspect("equal", "box")
        # self.canvas.ax.set_xlim(0, output.surf.x_max)
        # self.canvas.ax.set_ylim(0, output.surf.y_max)
        # self.canvas.fig.savefig(svg_buffer, format="svg", bbox_inches="tight")
        # self.canvas.draw()

        svg_buffer = io.BytesIO()
        output.svgplot_covered_grid(filename=svg_buffer)
        svg_data = svg_buffer.getvalue()

        self.svg_widget.load(svg_data)
        self.svg_widget.renderer().setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)

    def _save_settings_json(self) -> None:
        """Save settings to JSON file."""

    def _load_settings_json(self) -> None:
        """Load settings from JSON file."""

    def _orient_molecule(self) -> None:
        """Orient the molecule with first_time_loader."""

    def error(self, msg: str) -> None:
        """Handle the errors.

        :param msg: Error message to display in a new window.
        """
        QMessageBox.critical(self, "Input Error", msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = AdsorpyGUI()
    gui.resize(900, 500)
    gui.show()
    sys.exit(app.exec())
