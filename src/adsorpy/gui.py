# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""GUI module of adsorpy."""  # TODO: Make a new repo for this!
import io
import sys
import webbrowser

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QRegularExpression, Qt
from PySide6.QtGui import QAction, QDoubleValidator, QIcon, QIntValidator, QRegularExpressionValidator
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from shapely import MultiPolygon
from shapely.plotting import plot_polygon

from src.adsorpy import molecule_lib
from src.adsorpy.run_simulation import run_simulation, show_surface


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


        central = SurfaceGeneration()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout()
        central.setLayout(main_layout)

        controls_layout = QVBoxLayout()
        main_layout.addLayout(controls_layout)



class SurfaceGeneration(QWidget):
    """Surface generation window."""

    def __init__(self) -> None:
        """Initialise surface generation window."""
        super().__init__()
        self.setWindowTitle("Surface Generation")

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

        # Molecule dropdown
        controls_layout.addWidget(QLabel("Molecule:"))
        self.molecule_dropdown = QComboBox()
        self.molecule_names = list_public_molecules()
        self.molecule_dropdown.addItems(self.molecule_names)
        controls_layout.addWidget(self.molecule_dropdown)

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

        # Step limit input box
        controls_layout.addWidget(QLabel("Step limit (optional, > 0 int):"))
        self.step_limit = QLineEdit()
        self.step_limit.setPlaceholderText("e.g. 1")
        self.step_limit.setValidator(gt_one_validator)
        controls_layout.addWidget(self.step_limit)

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
