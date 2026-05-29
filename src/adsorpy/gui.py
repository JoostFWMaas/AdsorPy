# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""GUI module of adsorpy."""
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk
from shapely import MultiPolygon
from shapely.plotting import plot_polygon

from src.adsorpy import molecule_lib
from src.adsorpy.run_simulation import run_simulation


def list_public_molecules() -> list[str]:
    """List all molecules in molecule_lib.

    :returns: The molecules list, sorted.
    """
    molecules: list[str] = []
    for name in dir(molecule_lib):
        if name.startswith("_"):
            continue  # skip private members

        attr = getattr(molecule_lib, name)

        # Keep only objects that look like molecule definitions
        # Adjust this check depending on AdsorPy's class structure
        if callable(attr) and attr.__module__ == "adsorpy.molecule_lib":
            molecules.append(name)

    return sorted(molecules)

class AdsorPyGUI(tk.Tk):
    """Adsorpy GUI."""

    def __init__(self) -> None:
        """Initialize the GUI."""
        super().__init__()

        self.title("AdsorPy GUI")
        self.geometry("900x600")

        self._build_controls()
        self._build_plot()

        menu_bar = tk.Menu(self)
        self.config(menu=menu_bar)

        settings_menu = tk.Menu(menu_bar, tearoff=False)
        help_menu = tk.Menu(menu_bar, tearoff=False)
        iconsize = (
            15,  # Width
            15,  # Height
        )

        gear_icon = Image.open(Path("assets/gear-icon.png")).resize(iconsize)
        exit_icon = Image.open(Path("assets/exit-icon.png")).resize(iconsize)
        book_icon = Image.open(Path("assets/book-icon.png")).resize(iconsize)
        self.gear_icon = ImageTk.PhotoImage(gear_icon)
        self.exit_icon = ImageTk.PhotoImage(exit_icon)
        self.book_icon = ImageTk.PhotoImage(book_icon)



        settings_menu.add_command(
            label="Settings",
            command=self.destroy,
            image=self.gear_icon,
            compound="left",
        )

        settings_menu.add_command(
            label="Exit",
            command=self.destroy,
            image=self.exit_icon,
            compound="left",
        )

        menu_bar.add_cascade(
            label="File",
            menu=settings_menu,
            underline=0,
            # image=self.item_gear_icon,
            compound="left",
        )

        menu_bar.add_cascade(
            label="Help",
            menu=help_menu,
            underline=0,
            compound="left",
        )

        def open_documentation() -> None:
            """Open link to auto-generated documentation."""
            webbrowser.open("https://joostfwmaas.github.io/AdsorPy/")

        def open_wiki() -> None:
            """Open link to adsorpy wiki."""
            webbrowser.open("https://github.com/JoostFWMaas/AdsorPy/wiki")

        def report_bug() -> None:
            """Open link to issues page."""
            webbrowser.open("https://github.com/JoostFWMaas/AdsorPy/issues")

        help_menu.add_command(
            label="Documentation (web)",
            command=open_documentation,
            compound="left",
            image=self.book_icon,
        )

        help_menu.add_command(
            label="Wiki (web)",
            command=open_wiki,
            compound="left",
            image=self.book_icon,
        )

        help_menu.add_command(
            label="Report bug (web)",
            command=report_bug,
            compound="left",
            image=self.book_icon,
        )





        # help_menu.add_command(
        #     label="About",
        #     command=self.about,
        #     compound="left",
        # )


        # help_menu = tk.Menu(menu_bar, name="help")
        # menu_bar.add_cascade(label="Help", menu=help_menu, underline=0)
        # self.createcommand("tk::mac::ShowHelp",)


    def _build_controls(self) -> None:
        """Build the control frame."""
        frame = ttk.Frame(self, padding=10)
        frame.pack(side=tk.LEFT, fill=tk.Y)

        # # Molecule string (from molecule_lib / first_time_loader)
        # ttk.Label(frame, text="Molecule string:").pack(anchor="w")
        # self.molecule_entry = tk.Text(frame, height=5, width=40)
        # self.molecule_entry.pack(fill=tk.X, pady=(0, 10))

        # Number of molecules / steps
        ttk.Label(frame, text="Step limit:").pack(anchor="w")
        self.step_limit = tk.IntVar(value=10000)
        ttk.Entry(frame, textvariable=self.step_limit, width=15).pack(anchor="w", pady=(0, 10))

        # Random seed
        ttk.Label(frame, text="Random seed (optional):").pack(anchor="w")
        self.seed_var = (tk.StringVar(value=""))
        ttk.Entry(frame, textvariable=self.seed_var, width=15).pack(anchor="w", pady=(0, 10))

        # Surface / lattice type
        ttk.Label(frame, text="Surface type:").pack(anchor="w")
        # self.surface_var = tk.StringVar(value="hex_al2o3")
        # ttk.Entry(frame, textvariable=self.surface_var, width=20).pack(anchor="w", pady=(0, 10))

        self.surface_names = ["hexagonal", "honeycomb", "square"]
        self.surface_var = tk.StringVar(value=self.surface_names[0])

        self.surface_dropdown = ttk.Combobox(
            frame,
            textvariable=self.surface_var,
            values=self.surface_names,
            state="readonly",
            width=30,
        )

        self.surface_dropdown.pack(anchor="w", pady=(0, 15))

        # Run button
        ttk.Button(frame, text="Run simulation", command=self.run_simulation_clicked).pack(
            anchor="center", pady=20,
        )

        # Molecule dropdown
        ttk.Label(frame, text="Select molecule:").pack(anchor="w")

        self.molecule_names = list_public_molecules()
        self.molecule_var = tk.StringVar(value=self.molecule_names[0])

        self.molecule_dropdown = ttk.Combobox(
            frame,
            textvariable=self.molecule_var,
            values=self.molecule_names,
            state="readonly",
            width=30,
        )

        self.molecule_dropdown.pack(anchor="w", pady=(0, 15))

        # Status label
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(frame, textvariable=self.status_var, foreground="gray").pack(anchor="w", pady=(20, 0))

    def _build_plot(self) -> None:
        """Build the plot frame."""
        plot_frame = ttk.Frame(self, padding=10)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(5, 5), dpi=75)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Adsorbed molecules")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _save_settings_json(self) -> None:
        """Save settings to JSON file."""

    def _load_settings_json(self) -> None:
        """Load settings from JSON file."""

    def _orient_molecule(self) -> None:
        """Orient the molecule with first_time_loader."""

    def run_simulation_clicked(self) -> None:
        """To be added."""
        try:
            try:
                step_limit = int(self.step_limit.get())
            except ValueError:
                messagebox.showerror("Error", "Number of molecules must be an integer.")
                return


            # surface = self.surface_var.get().strip()

            self.status_var.set("Running simulation...")
            self.update_idletasks()

            # ------------------------------------------------------------------
            # 1) Convert molecule string to footprint using AdsorPy
            #    (adapt this to how you normally load molecules)
            #
            # Example pattern (check AdsorPy docs for exact API):
            #   molecule = molecule_lib.Molecule.from_string(molecule_str)
            #   footprint = molecule.get_footprint(surface=surface)
            #
            # Here we just keep a placeholder:
            # ------------------------------------------------------------------
            # TODO: replace with real molecule loading

            # ------------------------------------------------------------------
            # 2) Call AdsorPy's run_simulation
            #
            # Check the actual signature in AdsorPy. A common pattern might be:
            #   result = run_simulation(
            #       molecule_footprints=[footprint],
            #       n_molecules=n_mol,
            #       rng_seed=seed,
            #       surface=surface,
            #       # other keyword args...
            #   )
            #
            # We'll assume it returns coordinates or an object with them.
            # ------------------------------------------------------------------

            # TODO: replace this with the real call
            # result = run_simulation(
            #     molecule_footprints=[footprint],
            #     n_molecules=n_mol,
            #     rng_seed=seed,
            #     surface=surface,
            # )

            # For now, fake some data so the GUI structure is clear:
            seed = int(self.seed_var.get()) if self.seed_var.get() != "" else None

            output = run_simulation(timestep_limit=step_limit, seed=seed)[-1]

            # If you have real result, adapt here:
            # x, y = result.x_coords, result.y_coords



            # ------------------------------------------------------------------
            # 3) Plot result
            # ------------------------------------------------------------------
            self.ax.clear()
            # self.ax.scatter(x, y, s=5)
            plot_polygon(MultiPolygon(output.mol_data.stored_mirr_data["polygon"]), ax=self.ax, add_points=False)
            self.ax.set_title("Adsorbed molecules")
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")
            self.ax.set_aspect("equal", "box")
            self.ax.set_xlim(0, output.surf.x_max)
            self.ax.set_ylim(0, output.surf.y_max)

            self.canvas.draw()

            self.status_var.set("Done.")

        except tk.TclError as e:
            self.status_var.set("Error.")
            messagebox.showerror("Error during simulation", str(e))


if __name__ == "__main__":
    app = AdsorPyGUI()
    app.mainloop()
