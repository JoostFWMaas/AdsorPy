# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""GUI module of adsorpy."""  # TODO: Make a new repo for this!
import inspect
import io
import re
import sys
import webbrowser
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, ClassVar

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QRegularExpression, QSize, Qt, Signal
from PySide6.QtGui import QAction, QDoubleValidator, QIcon, QIntValidator, QRegularExpressionValidator
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from shapely import MultiPolygon, Polygon
from shapely.plotting import plot_polygon

from src.adsorpy import molecule_lib
from src.adsorpy.run_simulation import run_simulation, show_surface

if TYPE_CHECKING:
    from collections.abc import Callable


def list_public_molecules() -> list[str]:
    """List all molecules in molecule_lib.

    :returns: The molecules list, sorted.
    """
    molecules: list[str] = []

    for name in dir(molecule_lib):
        if name.startswith("_"):
            continue  # skip private members

        attr: object = getattr(molecule_lib, name)

        # Keep only objects that look like molecule definitions
        if callable(attr) and attr.__module__ == "adsorpy.molecule_lib":
            molecules.append(name)

    return sorted(molecules)

def extract_param_docs(func: Callable) -> dict[str, str]:
    doc = inspect.getdoc(func)
    if not doc:
        return {}

    param_docs: dict[str, str] = {}
    lines = doc.splitlines()

    current_param = None
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


class AutoStateMeta(type(QObject)):
    def __new__(cls, name, bases, attrs):
        fields = attrs.get("fields", {})

        for field_name, field_type in fields.items():
            signal_name = f"{field_name}Changed"
            private_name = f"_{field_name}"

            attrs[signal_name] = Signal(field_type)

            def getter(self, pn=private_name):
                return getattr(self, pn)

            def setter(self, value, pn=private_name, sn=signal_name):
                setattr(self, pn, value)
                getattr(self, sn).emit(value)

            attrs[field_name] = property(getter, setter)

        return super().__new__(cls, name, bases, attrs)

    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)

        # Auto‑initialize private fields
        for field_name in cls.fields:
            private_name = f"_{field_name}"
            if not hasattr(obj, private_name):
                setattr(obj, private_name, None)

        return obj


class AppState(QObject, metaclass=AutoStateMeta):
    """AppState class to communicate between tabs."""

    fields: ClassVar[dict[str, type]] = {
        "filepathm": str,
        "seed_input": QLineEdit,
        "count": int,
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
        pos_float_validator = QDoubleValidator(bottom=0)
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
        controls_layout.addWidget(self.step_limit)
        main_layout.addLayout(controls_layout)

        self.canvas = MplCanvas()
        svg_buffer = io.BytesIO()
        self.canvas.fig.savefig(svg_buffer, format="svg", bbox_inches="tight")
        svg_data = svg_buffer.getvalue()
        self.svg_widget = QSvgWidget()
        self.svg_widget.renderer().setAspectRatioMode(Qt.KeepAspectRatio)
        self.svg_widget.load(svg_data)

        # Add the SVG widget to layout
        # layout.addWidget(svg_widget)
        # main_layout.addWidget(self.canvas, stretch=1)
        main_layout.addWidget(self.svg_widget, stretch=1)

        self.setLayout(main_layout)


class MoleculeGeneration(QWidget):
    """Molecule generation widget."""

    def __init__(self, state: AppState) -> None:
        """Initialise the molecule generation widget.

        :param state: AppState shared state between tabs.
        """
        super().__init__()
        self.state = state

        main_layout = QHBoxLayout()
        controls_layout = QVBoxLayout()
        gt_one_validator = QIntValidator(bottom=1)
        pos_float_validator = QDoubleValidator(bottom=0)
        seed_validator = QRegularExpressionValidator(regularExpression=QRegularExpression(r"^\d+$"))


        # Add molecule button
        self.add_molecule_button = QPushButton("Add new molecule")
        # self.add_molecule_button.clicked.connect(self.add_molecule)
        controls_layout.addWidget(self.add_molecule_button)
        main_layout.addLayout(controls_layout)

        self.func_dropdown = QComboBox()
        controls_layout.addWidget(QLabel("Select molecule generator:"))
        controls_layout.addWidget(self.func_dropdown)

        temp_generators = {
            name: func for name, func in molecule_lib.__dict__.items() if callable(func) and not name.startswith("_") and func.__module__ == "adsorpy.molecule_lib"
        }

        # self.generators = OrderedDict(sorted(temp_generators.items(), key=lambda item: item[0]))
        self.generators: OrderedDict[str, Callable[[...], Polygon]] = OrderedDict(sorted(temp_generators.items()))

        self.func_dropdown.addItems(self.generators.keys())
        self.func_dropdown.currentTextChanged.connect(self.build_param_inputs)

        self.param_widgets = {}
        self.param_layout = QVBoxLayout()
        controls_layout.addLayout(self.param_layout)

        # Molecule dropdown
        controls_layout.addWidget(QLabel("Molecule:"))
        self.molecule_dropdown = QComboBox()
        self.molecule_names = list_public_molecules()
        self.molecule_dropdown.addItems(self.molecule_names)
        controls_layout.addWidget(self.molecule_dropdown)


        # Plot molecules
        self.canvas = MplCanvas()
        svg_buffer = io.BytesIO()
        self.canvas.fig.savefig(svg_buffer, format="svg", bbox_inches="tight")
        svg_data = svg_buffer.getvalue()
        self.svg_widget = QSvgWidget()
        self.svg_widget.renderer().setAspectRatioMode(Qt.KeepAspectRatio)
        self.svg_widget.load(svg_data)

        # Add the SVG widget to layout
        # layout.addWidget(svg_widget)
        # main_layout.addWidget(self.canvas, stretch=1)
        main_layout.addWidget(self.svg_widget, stretch=1)

        self.setLayout(main_layout)

        self.output_label = QLabel("")
        controls_layout.addWidget(self.output_label)
        main_layout.addLayout(controls_layout)

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
        main_layout.addLayout(right_col)

        # Build initial parameter UI
        self.build_param_inputs(self.func_dropdown.currentText())

        # Internal molecule storage
        self.molecules = []

    # ---------------------------------------------------------
    # Build parameter widgets dynamically
    # ---------------------------------------------------------
    def build_param_inputs(self, func_name: str) -> None:
        """Build the parameter inputs.

        :param func_name: Name of the function
        """
        while self.param_layout.count():
            item = self.param_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
            # if it's a nested layout, clear it too
            child_layout = item.layout()
            if child_layout is not None:
                while child_layout.count():
                    child_item = child_layout.takeAt(0)
                    cw = child_item.widget()
                    if cw is not None:
                        cw.deleteLater()

        func = self.generators[func_name]
        sig = inspect.signature(func)

        self.param_widgets = {}

        for name, param in sig.parameters.items():
            default = param.default
            is_optional: bool = default is None
            is_required: bool = param.default is inspect._empty
            param_docs = extract_param_docs(param)
            label = QLabel(name)

            # Tooltip on hover
            if name in param_docs:
                label.setToolTip(param_docs[name])

            row = QHBoxLayout()

            # Checkbox for optional parameters
            if is_optional:
                opt_checkbox = QCheckBox()
                opt_checkbox.setChecked(False)
                row.addWidget(opt_checkbox)
            else:
                opt_checkbox = None  # required argument

            if name in param_docs:
                info_btn = QToolButton()
                info_btn.setIcon(QIcon.fromTheme(QIcon.ThemeIcon.DialogInformation))
                info_btn.setIconSize(QSize(16, 16))
                info_btn.setToolTip("Show parameter help")

                def show_info(_, text=param_docs[name], pname=name):
                    QMessageBox.information(self, f"Parameter: {pname}", text)

                info_btn.clicked.connect(show_info)
                row.addWidget(info_btn)

            row.addWidget(QLabel(name))

            # Choose widget type based on default value
            # if isinstance(default, float) or param.annotation == "float":
            if param.annotation == "float":
                widget = QDoubleSpinBox(maximum=999, minimum=-999)
                widget.setDecimals(4)
                if not is_required:
                    widget.setValue(default)
            elif param.annotation == "PositiveFloat":
                widget = QDoubleSpinBox(maximum=999, minimum=0)
                widget.setDecimals(4)
                if not is_required:
                    widget.setValue(default)
            # elif isinstance(default, int) or param.annotation == "int":
            elif param.annotation == "int":
                widget = QSpinBox(maximum=999, minimum=-999)
                if not is_required:
                    widget.setValue(default)
            elif param.annotation == "PositiveInt":
                widget = QSpinBox(maximum=999, minimum=0)
                if not is_required:
                    widget.setValue(default)

            elif default is None:
                widget = QLineEdit()
                widget.setPlaceholderText("None")
            else:
                widget = QComboBox()
                widget.addItem(str(default))

            if is_optional:
                widget.setEnabled(False)
                opt_checkbox.toggled.connect(widget.setEnabled)

            row.addWidget(widget)
            self.param_widgets[name] = widget
            self.param_layout.addLayout(row)

    def add_molecule(self):
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
    def delete_molecule(self):
        idx = self.molecule_dropdown.currentIndex()
        if idx < 0:
            return

        # Remove from internal list
        del self.molecules[idx]

        # Remove from dropdown
        self.molecule_dropdown.removeItem(idx)

        self.output_label.setText("Molecule deleted")



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
        seed_validator = QRegularExpressionValidator(regularExpression=QRegularExpression(r"^\d+$"))

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
        self.svg_widget = QSvgWidget()
        self.svg_widget.renderer().setAspectRatioMode(Qt.KeepAspectRatio)
        self.svg_widget.load(svg_data)

        # Add the SVG widget to layout
        # layout.addWidget(svg_widget)
        # main_layout.addWidget(self.canvas, stretch=1)
        main_layout.addWidget(self.svg_widget, stretch=1)

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
        self.canvas.ax.clear()
        # self.canvas.ax.set_title("Adsorbed molecules")
        self.canvas.ax = show_surface(
            ax = self.canvas.ax,
            lattice_a=lattice,
            lattice_type=lattice_type,
            seed=seed,
            site_count=self.surface_count,
        )
        self.canvas.ax.set_xlabel("x")
        self.canvas.ax.set_ylabel("y")
        self.canvas.ax.set_aspect("equal", "box")

        svg_buffer = io.BytesIO()
        self.canvas.fig.savefig(svg_buffer, format="svg", bbox_inches="tight")
        svg_data = svg_buffer.getvalue()

        self.svg_widget.load(svg_data)
        self.svg_widget.renderer().setAspectRatioMode(Qt.KeepAspectRatio)
        # Add the SVG widget to layout
        # self.canvas.addWidget(svg_widget)

    def run_simulation(self) -> None:
        """Run the simulation."""
        # Validate seed
        seed_text = self.seed_input.text().strip()
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

        # Validate step limit
        step_text = self.step_limit.text().strip()
        step_limit: int | None = None
        if step_text:
            if not step_text.isdigit() or int(step_text) <= 0:
                self.error("Step limit must be a positive integer")
                return
            step_limit = int(step_text)

        molecule = self.molecule_dropdown.currentText()
        lattice_type = self.surface_dropdown.currentText()

        # Run simulation
        output = run_simulation(
            seed=seed,
            lattice_type=lattice_type,
            lattice_a=lattice,
            site_count=self.surface_count,
        )[-1]

        # ------------------------------------------------------
        # Plot result
        # ------------------------------------------------------
        self.canvas.ax.clear()
        plot_polygon(MultiPolygon(output.mol_data.stored_mirr_data["polygon"]), ax=self.canvas.ax, add_points=False)
        # self.canvas.ax.set_title("Adsorbed molecules")
        self.canvas.ax.set_xlabel("x")
        self.canvas.ax.set_ylabel("y")
        self.canvas.ax.set_aspect("equal", "box")
        self.canvas.ax.set_xlim(0, output.surf.x_max)
        self.canvas.ax.set_ylim(0, output.surf.y_max)

        svg_buffer = io.BytesIO()
        self.canvas.fig.savefig(svg_buffer, format="svg", bbox_inches="tight")
        svg_data = svg_buffer.getvalue()

        self.svg_widget.load(svg_data)
        self.svg_widget.renderer().setAspectRatioMode(Qt.KeepAspectRatio)

        # self.canvas.draw()

    def _save_settings_json(self) -> None:
        """Save settings to JSON file."""

    def _load_settings_json(self) -> None:
        """Load settings from JSON file."""

    def _orient_molecule(self) -> None:
        """Orient the molecule with first_time_loader."""

    def error(self, msg: str) -> None:
        """Handle the errors."""
        QMessageBox.critical(self, "Input Error", msg)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = AdsorpyGUI()
    gui.resize(900, 500)
    gui.show()
    sys.exit(app.exec())
