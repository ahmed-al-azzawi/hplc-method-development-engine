import streamlit as st
from Engine import Efficiency
from Engine import safe_read
from types import SimpleNamespace as sns
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import io


def test_get_safe_last():
    # Initialize mock session state lists
    st.session_state['all_backpressures'] = []
    st.session_state['all_resolutions'] = [1.32, 1.45, 1.0, 0.53, 0.67]

    # Test 1: Populated list returns the last element
    assert Efficiency._get_safe_last('all_resolutions') == 0.67

    # Test 2: Empty list returns default (None)
    assert Efficiency._get_safe_last('all_backpressures') is None

    # Test 3: Uninitialized/missing key returns default (None)
    assert Efficiency._get_safe_last('non_existent_key') is None

    # Test 4: Custom default fallback
    assert Efficiency._get_safe_last('non_existent_key', default=1000.0) == 1000.0

def test_find_efficiency_csv():
    st.session_state.clear()

    # Use pathlib to handle spaces and escaped paths cleanly
    csv_path = Path(
        "/Users/ahmedal-azzawi/Documents/Pharmaceutical Science/"
        "Second Year/Analytical methods 2/Test CSV/"
        "GR09-05-3-ACN-3-ISO-90-1.5-20-CP-0.1-C18aqP-150x4.6x5-amb-254 (1).csv"
    )

    if not csv_path.exists():
        pytest.skip(f"Test CSV file not found at {csv_path}")

    # Load raw bytes into an in-memory buffer that supports .getvalue()
    with open(csv_path, 'rb') as f:
        file_buffer = io.BytesIO(f.read())

    # file_buffer now behaves identically to st.file_uploader
    df = safe_read(file_buffer)
    result = Efficiency.find_efficiency(df)
    print(f''' 
    Plates: {result.plates}. Height: {result.height}.
    Appended plates: {st.session_state.plates[-1]}. Appended height: {st.session_state.heights[-1]}.
    ''')
    # Correct attribute names: result.plates and result.height
    assert result.plates == 1900
    assert result.height == 78.951

    # Session State assertions
    assert st.session_state.plates[-1] == 1900
    assert st.session_state.heights[-1] == 78.951

def test_predict_required_plates_standard():
    """Validates required plates calculation against known values and session history appending."""
    st.session_state.clear()

    # Inputs: Rs_current = 1.0, N_current = 10,000, Rs_spec = 1.5
    st.session_state['all_resolutions_final'] = [1.0]
    st.session_state['plates'] = [10_000]
    st.session_state['res_spec'] = 1.5

    # Calculation: ((1.5 * sqrt(10000)) / 1.0)^2 = (1.5 * 100)^2 = 150^2 = 22,500
    expected_plates = 22500.0

    result = Efficiency.predict_required_plates()

    # 1. Verify calculated value
    assert np.isclose(result, expected_plates)

    # 2. Verify state history tracking
    assert 'required_plates_history' in st.session_state
    assert st.session_state.required_plates_history[-1] == result


def test_predict_required_plates_fallback():
    """Validates behavior when session state is completely empty (uses defaults)."""
    st.session_state.clear()

    # Defaults: current_res=1.0, current_plates=10000, res_spec=1.5
    # Expected: ((1.5 * 100) / 1.0)^2 = 22,500
    result = Efficiency.predict_required_plates()

    assert np.isclose(result, 22500.0)
    assert st.session_state.required_plates_history == [22500.0]

def test_column_length_sufficient():
    """Test 1: Valid length found that meets required plates and satisfies pressure/time specs."""
    st.session_state.clear()

    # --- INPUT CONDITIONS ---
    # L_current = 100.0 mm, H = 10.0 µm = 0.01 mm
    # N_current = 100/0.01 = 10,000 plates
    st.session_state['all_column_lengths'] = [100.0]
    st.session_state['heights'] = [10.0]  # µm -> 0.01 mm
    st.session_state['all_backpressures'] = [1500.0]  # psi
    st.session_state['all_run_times'] = [10.0]  # min
    st.session_state['required_plates_history'] = [12000.0]  # Needs >12,000 plates

    # Specifications
    st.session_state['bpspec'] = 4000.0
    st.session_state['runspec'] = 20.0
    st.session_state['available_column_lengths'] = [30.0, 50.0, 100.0, 150.0, 250.0]

    # --- EXPECTED CALCULATIONS ---
    # current_height_mm = 10.0 / 1000.0 = 0.01 mm
    # Plate counts for available_lengths:
    # 30mm -> 3000, 50mm -> 5000, 100mm -> 10000, 150mm -> 15000, 250mm -> 25000
    # Valid candidates (N >= 12000): (150mm, 15000) and (250mm, 25000)
    # Selected min candidate: L = 150.0 mm, N = 15000.0
    # length_ratio = 150.0 / 100.0 = 1.5
    # predicted_bp = 1500.0 * 1.5 = 2250.0 psi  (<= 4000.0 -> OK)
    # predicted_tR = 10.0 * 1.5 = 15.0 min       (<= 20.0 -> OK)

    result = Efficiency.column_length()

    assert result is not None
    assert result['length'] == 150.0
    assert np.isclose(result['bp'], 2250.0)
    assert np.isclose(result['tR'], 15.0)
    assert np.isclose(result['plates'], 15000.0)


def test_column_length_exceeds_spec_returns_none():
    """Test 2: Valid length meets plates requirement, but breaches pressure or runtime spec."""
    st.session_state.clear()

    st.session_state['all_column_lengths'] = [100.0]
    st.session_state['heights'] = [10.0]  # 0.01 mm
    st.session_state['all_backpressures'] = [3000.0]  # psi
    st.session_state['all_run_times'] = [12.0]  # min
    st.session_state['required_plates_history'] = [12000.0]

    # Strict specifications
    st.session_state['bpspec'] = 4000.0  # Predicted BP (4500 psi) will breach this
    st.session_state['runspec'] = 15.0   # Predicted tR (18 min) will breach this
    st.session_state['available_column_lengths'] = [50.0, 100.0, 150.0, 250.0]

    # candidate L=150mm -> ratio=1.5 -> BP=4500 psi (> 4000) & tR=18 min (> 15)
    # enough_alone will evaluate to False -> Should return None
    result = Efficiency.column_length()

    assert result is None


def test_column_length_no_valid_candidates_returns_none():
    """Test 3: No available length provides enough theoretical plates."""
    st.session_state.clear()

    st.session_state['all_column_lengths'] = [50.0]
    st.session_state['heights'] = [10.0]  # 0.01 mm
    st.session_state['all_backpressures'] = [1000.0]
    st.session_state['all_run_times'] = [5.0]
    st.session_state['required_plates_history'] = [30000.0]  # Unusually high plate requirement

    # Max available column is 150mm -> max plates = 150 / 0.01 = 15000 (< 30000)
    st.session_state['available_column_lengths'] = [30.0, 50.0, 100.0, 150.0]

    result = Efficiency.column_length()

    assert result is None


def test_column_length_fallback_height_from_plates():
    """Test 4: Verifies fallback to `current_length / current_N` when `heights` is None."""
    st.session_state.clear()

    st.session_state['all_column_lengths'] = [100.0]
    st.session_state['heights'] = []  # No height history -> current_height is None
    st.session_state['plates'] = [10000]  # Fallback N -> current_height_mm = 100 / 10000 = 0.01 mm
    st.session_state['all_backpressures'] = [1000.0]
    st.session_state['all_run_times'] = [10.0]
    st.session_state['required_plates_history'] = [12000.0]

    st.session_state['bpspec'] = 4000.0
    st.session_state['runspec'] = 20.0
    st.session_state['available_column_lengths'] = [100.0, 150.0]

    result = Efficiency.column_length()

    assert result is not None
    assert result['length'] == 150.0
    assert np.isclose(result['bp'], 1500.0)
    assert np.isclose(result['tR'], 15.0)

def test_particle_size_sufficient():
    """Test 1: Valid particle size found that meets required plates and satisfies pressure/time specs."""
    st.session_state.clear()

    # --- INPUT CONDITIONS ---
    # Current state: dp1 = 5.0 µm, N1 = 10,000 plates, BP1 = 1000 psi, tR1 = 10.0 min
    st.session_state['all_particle_sizes'] = [5.0]
    st.session_state['plates'] = [10000]
    st.session_state['all_backpressures'] = [1000.0]
    st.session_state['all_run_times'] = [10.0]
    st.session_state['required_plates_history'] = [15000.0]  # Target: 15,000 plates

    # Specs
    st.session_state['bpspec'] = 5000.0
    st.session_state['runspec'] = 30.0
    st.session_state['available_particle_sizes'] = [1.8, 2.7, 3.5, 5.0]

    # --- EXPECTED CALCULATIONS ---
    # N_predicted = N1 * (dp1 / dp2)
    # 5.0 µm -> 10000 * (5.0/5.0) = 10000 (< 15000)
    # 3.5 µm -> 10000 * (5.0/3.5) = 14285.7 (< 15000)
    # 2.7 µm -> 10000 * (5.0/2.7) = 18518.5 (>= 15000) -> Valid!
    # 1.8 µm -> 10000 * (5.0/1.8) = 27777.8 (>= 15000) -> Valid!
    # Valid candidates: (2.7, 18518.5) and (1.8, 27777.8)
    # Selected max size: dp2 = 2.7 µm
    # size_ratio = (5.0 / 2.7)^2 = 3.429355
    # predicted_bp = 1000.0 * 3.429355 = 3429.355 psi (<= 5000.0 -> OK)
    # predicted_tR = 10.0 min (unchanged)

    result = Efficiency.particle_size()

    assert result is not None
    assert result['size'] == 2.7
    assert np.isclose(result['bp'], 3429.35528, atol=1e-3)
    assert np.isclose(result['tR'], 10.0)
    assert np.isclose(result['plates'], 18518.5185, atol=1e-3)


def test_particle_size_exceeds_bpspec_returns_none():
    """Test 2: Valid particle size delivers required plates, but inverse-square pressure increase exceeds bpspec."""
    st.session_state.clear()

    st.session_state['all_particle_sizes'] = [5.0]
    st.session_state['plates'] = [10000]
    st.session_state['all_backpressures'] = [2000.0]  # Higher starting BP
    st.session_state['all_run_times'] = [10.0]
    st.session_state['required_plates_history'] = [15000.0]

    # Strict pressure limit
    st.session_state['bpspec'] = 4000.0  # Predicted BP for 2.7 µm is 2000 * (5.0/2.7)^2 = 6858.7 psi
    st.session_state['runspec'] = 30.0
    st.session_state['available_particle_sizes'] = [1.8, 2.7, 3.5, 5.0]

    result = Efficiency.particle_size()

    assert result is None


def test_particle_size_no_valid_candidates_returns_none():
    """Test 3: No available particle size yields enough theoretical plates."""
    st.session_state.clear()

    st.session_state['all_particle_sizes'] = [5.0]
    st.session_state['plates'] = [5000]
    st.session_state['all_backpressures'] = [1000.0]
    st.session_state['all_run_times'] = [10.0]
    st.session_state['required_plates_history'] = [25000.0]  # Very high requirement

    # Max plates available with 1.8 µm: 5000 * (5.0 / 1.8) = 13888.8 (< 25000)
    st.session_state['available_particle_sizes'] = [1.8, 2.7, 3.5, 5.0]

    result = Efficiency.particle_size()

    assert result is None


def test_particle_size_fallback_predict_required_plates():
    """Test 4: Automatically predicts required plates when required_plates_history is missing."""
    st.session_state.clear()

    # Base parameters for required plates calculation:
    # default current_res = 1.0, current_plates = 10000, res_spec = 1.5 -> required_plates = 22500.0
    st.session_state['all_particle_sizes'] = [5.0]
    st.session_state['plates'] = [10000]
    st.session_state['all_backpressures'] = [500.0]
    st.session_state['all_run_times'] = [8.0]
    st.session_state['required_plates_history'] = None  # Forces fallback calculation

    st.session_state['bpspec'] = 5000.0
    st.session_state['runspec'] = 30.0
    st.session_state['available_particle_sizes'] = [1.8, 2.7, 3.5, 5.0]

    # Required plates = 22500.0
    # Candidate plate counts:
    # 2.7 µm -> 10000 * (5/2.7) = 18518.5 (< 22500)
    # 1.8 µm -> 10000 * (5/1.8) = 27777.8 (>= 22500) -> Valid!
    result = Efficiency.particle_size()

    assert result is not None
    assert result['size'] == 1.8
    assert np.isclose(result['bp'], 500.0 * (5.0 / 1.8) ** 2)
    assert np.isclose(result['plates'], 27777.777, atol=1e-3)


def test_spp_sufficient_conversion():
    """Test 1: Valid conversion from FPP to SPP yielding 50% plate increase and 15% pressure drop."""
    st.session_state.clear()

    # --- INPUT CONDITIONS ---
    # FPP column: N = 10,000, BP = 2000 psi, tR = 10.0 min, is_spp = False
    st.session_state['plates'] = [10000]
    st.session_state['all_backpressures'] = [2000.0]
    st.session_state['all_run_times'] = [10.0]
    st.session_state['is_spp_list'] = [False]
    st.session_state['required_plates_history'] = [14000.0]  # Target: 14,000 plates

    # Specs
    st.session_state['bpspec'] = 5000.0
    st.session_state['runspec'] = 30.0

    # --- EXPECTED CALCULATIONS ---
    # predicted_N = 10000 * 1.5 = 15000 (>= 14000 -> OK)
    # predicted_bp = 2000 * 0.85 = 1700 psi (<= 5000 -> OK)
    # predicted_tR = 10.0 min

    result = Efficiency.spp()

    assert result is not None
    assert result['is_spp'] is True
    assert np.isclose(result['plates'], 15000.0)
    assert np.isclose(result['bp'], 1700.0)
    assert np.isclose(result['tR'], 10.0)


def test_spp_already_spp_returns_none():
    """Test 2: Returns None immediately if column is already SPP."""
    st.session_state.clear()

    st.session_state['plates'] = [10000]
    st.session_state['all_backpressures'] = [1000.0]
    st.session_state['all_run_times'] = [10.0]
    st.session_state['is_spp_list'] = [True]  # Already SPP
    st.session_state['required_plates_history'] = [12000.0]

    result = Efficiency.spp()

    assert result is None


def test_spp_insufficient_plates_returns_none():
    """Test 3: Returns None if 1.5x plate boost does not meet required plates."""
    st.session_state.clear()

    st.session_state['plates'] = [10000]
    st.session_state['all_backpressures'] = [1000.0]
    st.session_state['all_run_times'] = [10.0]
    st.session_state['is_spp_list'] = [False]
    st.session_state['required_plates_history'] = [18000.0]  # Predicted 15,000 < 18,000

    result = Efficiency.spp()

    assert result is None


def test_spp_fallback_predict_required_plates():
    """Test 4: Verifies fallback to `predict_required_plates()` when required_plates_history is missing."""
    st.session_state.clear()

    # Define resolution state for predictable fallback calculation
    # Rs_current = 1.2, Rs_spec = 1.4 -> multiplier = (1.4 / 1.2)^2 = 1.3611
    # N_required = 16000 * 1.3611 = 21,777.7 plates
    st.session_state['all_resolutions_final'] = [1.2]
    st.session_state['res_spec'] = 1.4

    st.session_state['plates'] = [16000]  # Predicted N = 16000 * 1.5 = 24000 (>= 21778 -> OK)
    st.session_state['all_backpressures'] = [1000.0]
    st.session_state['all_run_times'] = [10.0]
    st.session_state['is_spp_list'] = [False]
    st.session_state['required_plates_history'] = None  # Triggers fallback

    st.session_state['bpspec'] = 5000.0
    st.session_state['runspec'] = 30.0

    result = Efficiency.spp()

    assert result is not None
    assert result['is_spp'] is True
    assert np.isclose(result['plates'], 24000.0)
    assert np.isclose(result['bp'], 850.0)

def test_flow_rate_all_specs_met_returns_none():
    """Test 1: All specs met (Rs >= 1.5, tR <= 15 min, BP <= 4000 psi) -> Returns None (No action needed)."""
    st.session_state.clear()

    st.session_state['all_flow_rates'] = [1.0]
    st.session_state['all_backpressures'] = [2000.0]
    st.session_state['all_run_times'] = [12.0]
    st.session_state['all_column_lengths'] = [150.0]
    st.session_state['all_particle_sizes'] = [5.0]
    st.session_state['is_spp_list'] = [False]
    st.session_state['all_resolutions_final'] = [1.6]  # >= 1.5 (In Spec)
    st.session_state['plates'] = [8000.0]

    st.session_state['bpspec'] = 4000.0
    st.session_state['runspec'] = 15.0
    st.session_state['res_spec'] = 1.5

    st.session_state['all_retention_times'] = [[1.0, 6.0, 12.0]]
    st.session_state['required_plates_history'] = [8000.0]

    result = Efficiency.flow_rate()

    assert result is None



def test_flow_rate_resolution_deficit_cannot_be_solved_returns_none():
    """Test 4: Rs < res_spec, but maximum possible proportional gain (~1.095x at u_opt) 

    yields N = 6023 plates < N_required (10000 plates) -> Returns None.
    Forces fallback to hardware scaling (L, dp, SPP).
    """
    st.session_state.clear()

    st.session_state['all_flow_rates'] = [1.0]
    st.session_state['all_backpressures'] = [2000.0]
    st.session_state['all_run_times'] = [12.0]
    st.session_state['all_column_lengths'] = [150.0]
    st.session_state['all_particle_sizes'] = [5.0]
    st.session_state['is_spp_list'] = [False]
    st.session_state['all_resolutions_final'] = [1.1]
    st.session_state['plates'] = [5500.0]

    st.session_state['bpspec'] = 4000.0
    st.session_state['runspec'] = 15.0
    st.session_state['res_spec'] = 1.5

    st.session_state['all_retention_times'] = [[1.0, 6.0, 12.0]]
    st.session_state['required_plates_history'] = [10000.0]

    result = Efficiency.flow_rate()

    assert result is None


def test_flow_rate_pressure_exceeded_returns_none():
    """Test 5: Time defect requires higher flow, but pressure exceeds bpspec (4000 psi) -> Returns None."""
    st.session_state.clear()

    st.session_state['all_flow_rates'] = [1.0]
    st.session_state['all_backpressures'] = [3800.0]  # High initial BP
    st.session_state['all_run_times'] = [20.0]        # > 15 min (Defect)
    st.session_state['all_column_lengths'] = [150.0]
    st.session_state['all_particle_sizes'] = [5.0]
    st.session_state['is_spp_list'] = [False]
    st.session_state['all_resolutions_final'] = [1.6]
    st.session_state['plates'] = [8000.0]

    st.session_state['bpspec'] = 4000.0
    st.session_state['runspec'] = 15.0
    st.session_state['res_spec'] = 1.5

    st.session_state['all_retention_times'] = [[1.0, 10.0, 20.0]]
    st.session_state['required_plates_history'] = [5000.0]

    result = Efficiency.flow_rate()

    assert result is None

def test_flow_rate_resolution_deficit_insufficient_returns_none():
    """Test: Rs = 1.20 (< 1.50 spec). Peak gamma ~ 1.095 yields max predicted Rs = 1.255.
    
    No flow rate can bridge 1.20 -> 1.50 -> Must return None to trigger hardware scaling.
    """
    st.session_state.clear()

    st.session_state['all_flow_rates'] = [1.0]
    st.session_state['all_backpressures'] = [2000.0]
    st.session_state['all_run_times'] = [12.0]
    st.session_state['all_column_lengths'] = [150.0]
    st.session_state['all_particle_sizes'] = [5.0]
    st.session_state['is_spp_list'] = [False]
    st.session_state['all_resolutions_final'] = [1.2]
    st.session_state['plates'] = [8000.0]

    st.session_state['bpspec'] = 4000.0
    st.session_state['runspec'] = 15.0
    st.session_state['res_spec'] = 1.5

    st.session_state['all_retention_times'] = [[1.0, 6.0, 12.0]]

    result = Efficiency.flow_rate()

    assert result is None


def test_flow_rate_resolution_deficit_solvable_direct_rs():
    """Test: Rs = 1.47 (< 1.50 spec). Small shift to F = 0.85 mL/min gives gamma = 1.0664.
    
    Predicted Rs = 1.47 * sqrt(1.0664) = 1.518 (>= 1.50 spec) -> VALID!
    """
    st.session_state.clear()

    st.session_state['all_flow_rates'] = [1.0]
    st.session_state['all_backpressures'] = [2000.0]
    st.session_state['all_run_times'] = [12.0]
    st.session_state['all_column_lengths'] = [150.0]
    st.session_state['all_particle_sizes'] = [5.0]
    st.session_state['is_spp_list'] = [False]
    st.session_state['all_resolutions_final'] = [1.47]
    st.session_state['plates'] = [8000.0]

    st.session_state['bpspec'] = 4000.0
    st.session_state['runspec'] = 15.0
    st.session_state['res_spec'] = 1.5

    st.session_state['all_retention_times'] = [[1.0, 6.0, 12.0]]

    result = Efficiency.flow_rate()

    assert result is not None
    assert np.isclose(result['flow_rate'], 0.85)

def test_calculate_rs_floor_standard_scaling():
    """Test 1: Standard headroom and resolution floor calculation.

    Validates exact scaling for HR = sqrt(3), N_max_fpp, N_max_spp, and Rs floors.
    """
    st.session_state.clear()

    st.session_state['all_backpressures'] = [2000.0]
    st.session_state['all_run_times'] = [10.0]
    st.session_state['all_column_lengths'] = [150.0]
    st.session_state['all_particle_sizes'] = [5.0]
    st.session_state['plates'] = [8000.0]

    st.session_state['bpspec'] = 4000.0
    st.session_state['runspec'] = 15.0
    st.session_state['res_spec'] = 1.5

    result = Efficiency.calculate_rs_floor()

    expected_hr = np.sqrt((4000.0 / 2000.0) * (15.0 / 10.0))  # sqrt(3) ~ 1.7320508
    h_min = 1.5 + 2.0 * np.sqrt(0.45)                        # ~ 2.8416408
    expected_n_max_fpp = (1000.0 * 150.0 * expected_hr) / (5.0 * h_min)
    expected_n_max_spp = expected_n_max_fpp * 1.5

    expected_rs_floor_fpp = 1.5 * np.sqrt(8000.0 / expected_n_max_fpp)
    expected_rs_floor_spp = 1.5 * np.sqrt(8000.0 / expected_n_max_spp)

    assert np.isclose(result['headroom'], expected_hr)
    assert np.isclose(result['n_max_fpp'], expected_n_max_fpp)
    assert np.isclose(result['n_max_spp'], expected_n_max_spp)
    assert np.isclose(result['rs_floor_fpp'], expected_rs_floor_fpp)
    assert np.isclose(result['rs_floor_spp'], expected_rs_floor_spp)


def test_calculate_rs_floor_zero_or_negative_operating_conditions():
    """Test 2: Edge case handling when backpressure or runtime is 0 or uninitialized.
    
    Ensures headroom defaults safely to 1.0 without throwing division errors.
    """
    st.session_state.clear()

    st.session_state['all_backpressures'] = [0.0]  # Zero BP edge case
    st.session_state['all_run_times'] = [10.0]
    st.session_state['all_column_lengths'] = [150.0]
    st.session_state['all_particle_sizes'] = [5.0]
    st.session_state['plates'] = [8000.0]

    st.session_state['bpspec'] = 4000.0
    st.session_state['runspec'] = 15.0
    st.session_state['res_spec'] = 1.5

    result = Efficiency.calculate_rs_floor()

    assert result['headroom'] == 1.0


def test_calculate_rs_floor_high_current_efficiency():
    """Test 3: When current_N exceeds N_max, Rs_floor exceeds res_spec.
    
    Verifies that the engine correctly reflects when hardware capacity is saturated.
    """
    st.session_state.clear()

    st.session_state['all_backpressures'] = [3800.0]  # Low headroom
    st.session_state['all_run_times'] = [14.5]
    st.session_state['all_column_lengths'] = [50.0]   # Short column -> low N_max
    st.session_state['all_particle_sizes'] = [5.0]
    st.session_state['plates'] = [25000.0]            # High current plates

    st.session_state['bpspec'] = 4000.0
    st.session_state['runspec'] = 15.0
    st.session_state['res_spec'] = 1.5

    result = Efficiency.calculate_rs_floor()

    assert result['rs_floor_fpp'] > 1.5
    assert result['rs_floor_spp'] == result['rs_floor_fpp'] / np.sqrt(1.5)

def test_diagnose_flow_failure_pressure_constrained():
    """Test 1: Backpressure at max flow (2.0 mL/min) exceeds bpspec (4000.0 psi).

    Math Verification:
    - Current: F = 1.2 mL/min, BP = 3000.0 psi
    - BP at 2.0 mL/min = 3000.0 * (2.0 / 1.2) = 5000.0 psi
    - 5000.0 psi > 4000.0 psi -> Returns pressure constraint message.
    """
    st.session_state.clear()

    st.session_state['all_backpressures'] = [3000.0]
    st.session_state['all_flow_rates'] = [1.2]
    st.session_state['bpspec'] = 4000.0

    result = Efficiency.diagnose_flow_failure()

    expected_msg = "Flow rate increase is constrained by pressure limit (at 2.0 mL/min, BP reaches 5000.0 psi vs 4000.0 psi spec)."
    assert result == expected_msg


def test_diagnose_flow_failure_kinetic_constrained():
    """Test 2: Backpressure at max flow (2.0 mL/min) is within bpspec.

    Math Verification:
    - Current: F = 1.0 mL/min, BP = 1800.0 psi
    - BP at 2.0 mL/min = 1800.0 * (2.0 / 1.0) = 3600.0 psi
    - 3600.0 psi <= 4000.0 psi -> Returns kinetic efficiency failure message.
    """
    st.session_state.clear()

    st.session_state['all_backpressures'] = [1800.0]
    st.session_state['all_flow_rates'] = [1.0]
    st.session_state['bpspec'] = 4000.0

    result = Efficiency.diagnose_flow_failure()

    expected_msg = "Flow rate adjustments alone cannot yield sufficient theoretical plates ($N$) to achieve required resolution."
    assert result == expected_msg


def test_diagnose_flow_failure_zero_current_flow_fallback():
    """Test 3: Edge case where current_flow is 0 or negative.

    Ensures zero-division safety check holds and falls back to comparing current_bp against bpspec.
    """
    st.session_state.clear()

    st.session_state['all_backpressures'] = [4500.0]
    st.session_state['all_flow_rates'] = [0.0]
    st.session_state['bpspec'] = 4000.0

    result = Efficiency.diagnose_flow_failure()

    expected_msg = "Flow rate increase is constrained by pressure limit (at 2.0 mL/min, BP reaches 4500.0 psi vs 4000.0 psi spec)."
    assert result == expected_msg



def main():
    test_get_safe_last()
    test_find_efficiency_csv()
    test_predict_required_plates_standard()
    test_predict_required_plates_fallback()
    test_column_length_exceeds_spec_returns_none()
    test_column_length_no_valid_candidates_returns_none()
    test_column_length_fallback_height_from_plates()
    test_particle_size_sufficient()
    test_particle_size_exceeds_bpspec_returns_none()
    test_particle_size_no_valid_candidates_returns_none()
    test_particle_size_fallback_predict_required_plates()
    test_spp_sufficient_conversion()
    test_spp_already_spp_returns_none()
    test_spp_insufficient_plates_returns_none()
    test_spp_fallback_predict_required_plates()
    test_flow_rate_all_specs_met_returns_none()
    test_flow_rate_resolution_deficit_cannot_be_solved_returns_none()
    test_flow_rate_resolution_deficit_insufficient_returns_none()
    test_flow_rate_pressure_exceeded_returns_none()
    test_calculate_rs_floor_standard_scaling()
    test_calculate_rs_floor_zero_or_negative_operating_conditions()
    test_calculate_rs_floor_high_current_efficiency()
    test_diagnose_flow_failure_pressure_constrained()
    test_diagnose_flow_failure_kinetic_constrained()
    test_diagnose_flow_failure_zero_current_flow_fallback()
    print("\033[1;33mALL TESTS PASSED!\033[0m")

if __name__ == '__main__':
    main()
