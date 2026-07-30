# Copyright (c) 2025-2026 Contributors to the AdsorPy project.
# SPDX-License-Identifier: MIT
"""Test the AppState and AutoStateMeta classes of the `gui.py` module."""
from PySide6.QtWidgets import QLineEdit
from pytestqt.qtbot import QtBot

from src.adsorpy.gui import AppState, GeneralSettings, SurfaceParameters


def test_app_state_metaclass_scans_annotations() -> None:
    """Verify AutoStateMeta successfully discovers typed field annotations at runtime."""
    # Ensure fields dictionary is correctly populated by the metaclass scanning loop
    assert "seed_input" in AppState.fields
    assert "surface_params" in AppState.fields
    assert "molecule_param_list" in AppState.fields
    assert "gap_size_distribution" in AppState.fields

    # Verify standard class exclusions worked (fields and private descriptors ignored)
    assert "fields" not in AppState.fields


def test_app_state_initializes_private_fields_to_none() -> None:
    """Verify the __call__ method populates missing private backend targets to None."""
    state = AppState()

    # Assert private fields are instantiated automatically
    assert hasattr(state, "_seed_input")
    assert hasattr(state, "_surface_params")
    assert state._seed_input is None
    assert state._surface_params is None


def test_setting_property_emits_changed_signal(qtbot: QtBot) -> None:
    """Verify that updating a state property fires its corresponding Changed signal.

    :param qtbot: QtBot instance to mock app interaction.
    """
    state = AppState()

    # Define a temporary payload tracking object
    mock_surface_data = SurfaceParameters(lattice_type="square", site_count=100)

    # Set up a pytest-qt signal blocker to catch the dynamically generated signal
    with qtbot.waitSignal(state.surface_paramsChanged, timeout=1000) as blocker:
        # Trigger the custom property setter
        state.surface_params = mock_surface_data

    # Assert that the data captured on the line loop matches the set object
    assert blocker.args == [mock_surface_data]
    assert state.surface_params == mock_surface_data


def test_widget_state_property_mutations(qtbot: QtBot) -> None:
    """Verify properties targeting QWidgets accept assignments and dispatch signals cleanly.

    :param qtbot: QtBot instance to mock app interaction.
    """
    state = AppState()

    # Instantiate actual UI components (must pass parent/register to qtbot if needed)
    temp_line_edit = QLineEdit()
    qtbot.addWidget(temp_line_edit)

    with qtbot.waitSignal(state.seed_inputChanged, timeout=1000) as blocker:
        state.seed_input = temp_line_edit

    assert blocker.args == [temp_line_edit]
    assert state.seed_input == temp_line_edit


def test_general_settings_listens_to_real_app_state(qtbot: QtBot) -> None:
    """Integration test confirming GeneralSettings receives updates from a real AppState instance.

    :param qtbot: QtBot instance to mock app interaction.
    """
    state = AppState()

    tab = GeneralSettings(state)
    qtbot.addWidget(tab)

    # Verify initial text-box conditions default correctly
    assert tab.initiated_surface_textbox.text() == "Default."
    assert tab.initiated_molecules_textbox.text() == "Default."

    # Mutate AppState directly from an external context (simulating another tab)
    temp_surf = SurfaceParameters(lattice_type="hexagonal", site_count=50)

    with qtbot.waitSignal(state.surface_paramsChanged, timeout=1000):
        state.surface_params = temp_surf

    # Assert that GeneralSettings reacted instantly to the event mutation hook
    assert tab.initiated_surface_textbox.text() == "User-defined."
