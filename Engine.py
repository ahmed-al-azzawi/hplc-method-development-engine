# ==============================================================================
# HPLC METHOD DEVELOPMENT ENGINE IN PYTHON
# ==============================================================================

import sys
from streamlit.web import cli as stcli
import streamlit as st
import pandas as pd
import chardet
import numpy as np
import io
from scipy.stats import linregress
from math import ceil
from plotly import graph_objects as go
import random
from scipy.optimize import linear_sum_assignment
from types import SimpleNamespace as sns 
import streamlit.components.v1 as components
import json
import traceback

st.set_page_config(
    page_title="HPLC Method Development App",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.session_state.pending_ligand_change = False
st.session_state.tr_slope = 0
st.session_state.tr_intercept = 0
st.session_state.tr_tzero = 0


def render_next_run_conditions_preview():
    """Displays a persistent, pre-filled preview card showing all committed 

    next-run parameters across Efficiency, Retention, and Selectivity cycles.
    """
    with st.expander('View Next Run Conditions'):
        st.text(
            'The next run\'s conditions are already saved in the session state. The user is not required to type them.'
            '\n You can view them here.'
        )
        # Verify that a next run has been committed (more parameter entries than executed runs)
        flow_rates = st.session_state.get('all_flow_rates', [])
        peak_counts = st.session_state.get('all_peak_counts', [])

        if len(flow_rates) > len(peak_counts) and len(flow_rates) > 0:
            # Safe extraction helper for last element
            def get_last(key, default="N/A"):
                arr = st.session_state.get(key, [])
                return arr[-1] if isinstance(arr, list) and len(arr) > 0 else default

            # 1. Efficiency Parameters
            next_L = get_last('all_column_lengths', 150.0)
            next_dp = get_last('all_particle_sizes', 5.0)
            next_is_spp = get_last('all_coreshells', False)
            next_spp_str = "Core-Shell (SPP)" if next_is_spp else "Fully Porous (FPP)"
            next_F = get_last('all_flow_rates', 1.0)
            next_id = get_last('all_internal_diameters', 4.6)

            # 2. Retention & Selectivity Parameters
            next_solvent = get_last('all_solvents', 'Acetonitrile')
            next_pct_b = get_last('all_pct_bs', 50.0)
            next_ligand = get_last('all_ligands', 'C18')
            next_temp = get_last('all_temperatures', 30.0)
            next_pore = get_last('all_pore_sizes', 100.0)

            # Card Container
            with st.container(border=True):
                st.markdown("### 📋 **Next Run Pre-Filled Conditions**")
                st.success("✅ **Conditions are already saved in the session state.**")
                st.caption(
                    "💡 *When a new file is uploaded, these conditions will automatically fill in the historical row.*"
                )
                st.divider()

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### ⚡ **Efficiency & Hardware**")
                    st.markdown(
                        f"- **Column Length ($L$):** `{next_L:.0f} mm`\n"
                        f"- **Particle Size ($d_p$):** `{next_dp:.1f} µm`\n"
                        f"- **Particle Tech:** `{next_spp_str}`\n"
                        f"- **Internal Diameter:** `{next_id:.1f} mm`\n"
                        f"- **Flow Rate ($F$):** `{next_F:.2f} mL/min`"
                    )

                with col2:
                    st.markdown("#### 🧪 **Retention & Selectivity**")
                    st.markdown(
                        f"- **Mobile Phase Solvent:** `{next_solvent}`\n"
                        f"- **Organic Modifier (%B):** `{next_pct_b:.1f}%`\n"
                        f"- **Stationary Phase Ligand:** `{next_ligand}`\n"
                        f"- **Column Temperature ($T$):** `{next_temp} °C`\n"
                        f"- **Pore Size:** `{next_pore:.0f} Å`"
                    )


class Efficiency:

    @staticmethod
    def _get_safe_last(key: str, default=None):
        val = st.session_state.get(key, [])
        if isinstance(val, list) and len(val) > 0:
            return val[-1]
        return default

    @staticmethod
    def find_efficiency(df: pd.DataFrame) -> sns:
        target_string = 'Peak Table'
        matches = df.astype(str).apply(lambda col: col.str.contains(target_string, case=False, na=False))
        if not matches.any().any():
            raise ValueError(f"'{target_string}' anchor was not found in the DataFrame.")
        
        match_locations = matches.stack()
        anchor_row, anchor_col = match_locations[match_locations].index[0]
        anchor_row_idx = df.index.get_loc(anchor_row)

        print(f"Anchor found at Row Index: {anchor_row_idx}")
        print(f'Column Index Anchor: {anchor_col}')
        
        plate_count_col = 12
        plate_height_col = 13

        # 2. Locate end anchor (first occurrence of 'Results' or 'Signal' AFTER start anchor)
        end_matches = df.iloc[anchor_row_idx + 1:].astype(str).apply(
            lambda col: col.str.contains('Results|Signal|Chromatogram', case=False, na=False)
        )

        # 3. Bound sub_df strictly between start and end anchors
        if end_matches.any().any():
            anchor_end = end_matches.stack()[end_matches.stack()].index[0][0]
            sub_df = df.iloc[anchor_row_idx + 1 : anchor_end]
        else:
            sub_df = df.iloc[anchor_row_idx + 1 :]
        valid_plates = pd.to_numeric(sub_df.iloc[:, plate_count_col], errors='coerce').dropna()
        valid_plates = valid_plates[valid_plates > 0]
        print(f'Valid Plates: {str(valid_plates)}')
        valid_plates = valid_plates[valid_plates.astype(str).str.strip() != '']
        if valid_plates.empty:
            raise ValueError("No valid plate number found in column 13 below 'Peak Table'.")
        plate_count = float(valid_plates.iloc[-1])
        
        valid_heights = sub_df.iloc[:, plate_height_col].dropna()
        valid_heights = valid_heights[valid_heights.astype(str).str.strip() != '']
        if valid_heights.empty:
            raise ValueError("No valid plate height found in column 14 below 'Peak Table'.")
        plate_height = float(valid_heights.iloc[-1])
        
        for val in ['plates', 'heights']:
            if val not in st.session_state or not isinstance(st.session_state[val], list):
                st.session_state[val] = []
        
        st.session_state.plates.append(int(plate_count))
        st.session_state.heights.append(plate_height)
        return sns(plates=int(plate_count), height=plate_height)

    @staticmethod
    def predict_required_plates():
        current_res = Efficiency._get_safe_last('all_resolutions_final', 1.0)
        current_plates = Efficiency._get_safe_last('plates', 10000)
        
        required_res = st.session_state.get('res_spec', 1.5)
        print(f'Current plates: {current_plates}. Current res: {current_res}. Required res: {required_res}')
        
        required_plates = ((required_res * (current_plates ** 0.5)) / current_res) ** 2
        if 'required_plates_history' not in st.session_state or not isinstance(st.session_state.required_plates_history, list):
            st.session_state.required_plates_history = [] 
            
        st.session_state.required_plates_history.append(required_plates)
        return required_plates

    @staticmethod
    def column_length(care_about_resolution=True):
        current_length = Efficiency._get_safe_last('all_column_lengths', 150.0)
        current_height = Efficiency._get_safe_last('heights', None)
        current_rs = Efficiency._get_safe_last('all_resolutions_final', 1.0)
        
        if current_height is None:
            current_N = Efficiency._get_safe_last('plates', 10000)
            current_height_mm = current_length / current_N
        else:
            current_height_mm = current_height / 1000.0

        current_bp = Efficiency._get_safe_last('all_backpressures', 1000.0)
        current_tR = Efficiency._get_safe_last('all_run_times', 10.0)
        required_plates = Efficiency._get_safe_last('required_plates_history', None)
        if required_plates is None:
            required_plates = Efficiency.predict_required_plates()

        available_lengths = st.session_state.get('available_column_lengths', [50.0, 100.0, 150.0, 250.0])
        plate_counts = [L / current_height_mm for L in available_lengths]
        valid_candidates = [(L, N) for L, N in zip(available_lengths, plate_counts) if N >= required_plates]

        if not valid_candidates:
            return None

        new_length, predicted_N = min(valid_candidates, key=lambda pair: pair[0])
        length_ratio = new_length / current_length
        predicted_bp = current_bp * length_ratio
        predicted_tR = current_tR * length_ratio

        enough_alone = (
            predicted_bp <= st.session_state.get('bpspec', 4000.0) and
            predicted_tR <= st.session_state.get('runspec', 15.0)
        )
        predicted_rs = current_rs * (length_ratio ** 0.5)
        print(f'Predicted Rs for COL LEN: {predicted_rs}')
        if care_about_resolution:
            res_eval = predicted_rs >= st.session_state.res_spec
        else:
            res_eval = True
        if enough_alone and res_eval:
            return {'length': new_length, 'bp': predicted_bp, 'tR': predicted_tR, 'plates': predicted_N, 'rs': predicted_rs,
                    'changes': [f'Column Length: {current_length} mm → {new_length}'], 'equipment': 'Column'}
        return None

    @staticmethod
    def particle_size():
        current_bp = Efficiency._get_safe_last('all_backpressures', 1000.0)
        current_tR = Efficiency._get_safe_last('all_run_times', 10.0)
        current_N = Efficiency._get_safe_last('plates', 10000)
        current_PS = Efficiency._get_safe_last('all_particle_sizes', 5.0)
        current_rs = Efficiency._get_safe_last('all_resolutions_final', 1.0)
        
        required_plates = Efficiency._get_safe_last('required_plates_history', None)
        if required_plates is None:
            required_plates = Efficiency.predict_required_plates()

        available_sizes = st.session_state.get('available_particle_sizes', [1.8, 2.7, 3.5, 5.0])
        plate_counts = [current_N * (current_PS / new_PS) for new_PS in available_sizes]
        valid_candidates = [(PS, N) for PS, N in zip(available_sizes, plate_counts) if N >= required_plates]

        if not valid_candidates:
            return None

        new_size, predicted_N = max(valid_candidates, key=lambda pair: pair[0])
        size_ratio = (current_PS / new_size) ** 2
        predicted_bp = current_bp * size_ratio
        predicted_tR = current_tR

        enough_alone = (
            predicted_bp <= st.session_state.get('bpspec', 5000.0) and
            predicted_tR <= st.session_state.get('runspec', 30.0)
        )
        predicted_rs = current_rs * ((predicted_N / current_N) ** 0.5)
        if enough_alone:
            return {'size': new_size, 'bp': predicted_bp, 'tR': predicted_tR, 'plates': predicted_N, 'rs':predicted_rs}
        return None

    @staticmethod
    def spp():
        current_bp = Efficiency._get_safe_last('all_backpressures', 1000.0)
        current_tR = Efficiency._get_safe_last('all_run_times', 10.0)
        current_N = Efficiency._get_safe_last('plates', 10000)
        current_is_spp = Efficiency._get_safe_last('all_coreshells', False)
        current_rs = Efficiency._get_safe_last('all_resolutions_final', 1.0)
        if current_is_spp:
            return None

        required_plates = Efficiency._get_safe_last('required_plates_history', None)
        if required_plates is None:
            required_plates = Efficiency.predict_required_plates()

        predicted_N = current_N * 1.5
        predicted_bp = current_bp
        predicted_tR = current_tR

        enough_alone = (
            predicted_N >= required_plates and
            predicted_bp <= st.session_state.get('bpspec', 5000.0) and
            predicted_tR <= st.session_state.get('runspec', 30.0)
        )
        print(f'N: {predicted_N}, BP: {predicted_bp}, tR: {predicted_tR}. Required: {required_plates}')
        predicted_rs = current_rs * ((predicted_N / current_N) ** 0.5)
        if enough_alone:
            return {'is_spp': True, 'bp': predicted_bp, 'tR': predicted_tR, 'plates': predicted_N, 'rs': predicted_rs}
        return None

    @staticmethod
    def flow_rate():
        current_flow = Efficiency._get_safe_last('all_flow_rates', 1.0)
        current_bp = Efficiency._get_safe_last('all_backpressures', 1000.0)
        current_tR = Efficiency._get_safe_last('all_run_times', 10.0)
        current_length = Efficiency._get_safe_last('all_column_lengths', 150.0)
        current_dp = Efficiency._get_safe_last('all_particle_sizes', 5.0)
        current_is_spp = Efficiency._get_safe_last('all_coreshells', False)
        current_rs = Efficiency._get_safe_last('all_resolutions_final', 1.0)
        current_N = Efficiency._get_safe_last('plates', 10000.0)
        
        bpspec = st.session_state.get('bpspec', 5000.0)
        runspec = st.session_state.get('runspec', 30.0)
        res_spec = st.session_state.get('res_spec', 1.5)

        ret_values = st.session_state.get('all_retention_times', [])
        if not ret_values or len(ret_values[-1]) == 0:
            return None
            
        all_tzeros = [tx[0] for tx in ret_values]
        tzero = float(np.mean(all_tzeros))

        # Calculate baseline linear velocity and model HETP at current flow rate
        u_1 = current_length / (tzero * 60.0)
        spp_factor = 1.5 if current_is_spp else 1.0
        hetp_1 = (1.5 * current_dp + (3.0 / u_1) + 0.15 * (current_dp ** 2) * u_1) / spp_factor

        # Identify method deficits
        rs_defect = current_rs < res_spec
        time_defect = current_tR > runspec

        # If method is fully in spec, no flow adjustment is needed
        if not rs_defect and not time_defect:
            return None

        flow_candidates = np.arange(0.3, 2.05, 0.05)
        valid_candidates = []

        for F_cand in flow_candidates:
            flow_ratio = F_cand / current_flow
            u_2 = u_1 * flow_ratio
            
            # Calculate candidate model HETP
            hetp_2 = (1.5 * current_dp + (3.0 / u_2) + 0.15 * (current_dp ** 2) * u_2) / spp_factor
            
            # Relative efficiency ratio (gamma) & direct resolution prediction
            gamma = hetp_1 / hetp_2
            predicted_rs = current_rs * np.sqrt(gamma)
            predicted_N = current_N * gamma
            
            predicted_bp = current_bp * flow_ratio
            predicted_tR = current_tR / flow_ratio

            # Enforce physical pressure and run-time boundary specs
            if predicted_bp <= bpspec and predicted_tR <= runspec:
                if rs_defect:
                    # Must hit or exceed resolution spec directly
                    if predicted_rs >= res_spec:
                        valid_candidates.append({
                            'flow_rate': float(round(F_cand, 2)),
                            'bp': float(predicted_bp),
                            'tR': float(predicted_tR),
                            'plates': float(predicted_N),
                            'predicted_rs': float(predicted_rs)
                        })
                elif time_defect:
                    # Must reduce run time below spec without dropping resolution below spec
                    if predicted_tR <= runspec and predicted_rs >= res_spec:
                        valid_candidates.append({
                            'flow_rate': float(round(F_cand, 2)),
                            'bp': float(predicted_bp),
                            'tR': float(predicted_tR),
                            'plates': float(predicted_N),
                            'predicted_rs': float(predicted_rs)
                        })

        if not valid_candidates:
            return None

        # Pick the highest flow candidate that meets specifications to optimize throughput
        best_candidate = max(valid_candidates, key=lambda x: x['flow_rate'])
        return {
            'flow_rate': best_candidate['flow_rate'],
            'bp': best_candidate['bp'],
            'tR': best_candidate['tR'],
            'plates': best_candidate['plates'],
            'rs': best_candidate['predicted_rs']
        }

    @staticmethod
    def calculate_rs_floor() -> dict:
        """Calculates absolute resolution floors matching baseline FPP vs SPP headroom."""
        current_bp = float(Efficiency._get_safe_last('all_backpressures', 1000.0))
        current_tR = float(Efficiency._get_safe_last('all_run_times', 10.0))
        current_L = float(Efficiency._get_safe_last('all_column_lengths', 150.0))
        current_PS = float(Efficiency._get_safe_last('all_particle_sizes', 5.0))
        current_dp = current_PS

        ret_values = st.session_state.get('all_retention_times', [])
        if not ret_values or len(ret_values[-1]) == 0:
            st.warning("No retention time data available for scaling.")
            return None

        all_tzeros = [tx[0] for tx in ret_values]
        tzero = float(np.mean(all_tzeros))

        # 1. Linear velocity u1 in mm/s
        u1 = current_L / (tzero * 60.0) if tzero > 0 else 0.0

        # 2. Baseline HETP and N_H_calc strictly for FPP (spp_factor = 1.0)
        hetp1_fpp = (1.5 * current_dp + (3.0 / u1) + 0.15 * (current_dp ** 2) * u1) if u1 > 0 else 1.0
        N_H_calc_fpp = (current_L * 1000.0) / hetp1_fpp

        bpspec = float(st.session_state.get('bpspec', 5000.0))
        runspec = float(st.session_state.get('runspec', 30.0))
        res_spec = float(st.session_state.get('res_spec', 1.5))

        # 3. Operational Headroom (HR)
        hr = np.sqrt((bpspec / current_bp) * (runspec / current_tR)) if (current_bp > 0 and current_tR > 0) else 1.0

        # 4. Van Deemter minimum reduced plate height factor: h_min = A + 2*sqrt(B*C) ~ 3.042
        van_deemter_min_h_factor = 1.5 + 2.0 * np.sqrt(3.0 * 0.15)

        # 5. Theoretical maximum plate counts (Nmax)
        n_max_fpp = (1000.0 * current_L * hr) / (current_PS * van_deemter_min_h_factor)
        n_max_spp = n_max_fpp * 1.5

        # 6. Resolution Floors
        rs_floor_fpp = res_spec * np.sqrt(N_H_calc_fpp / n_max_fpp) if n_max_fpp > 0 else res_spec
        rs_floor_spp = res_spec * np.sqrt(N_H_calc_fpp / n_max_spp) if n_max_spp > 0 else res_spec


        print(f'FPP: {rs_floor_fpp}')
        print(f'SPP: {rs_floor_spp}')

        return {
            'headroom': float(hr),
            'hetp_fpp': float(hetp1_fpp),
            'n_h_calc_fpp': float(N_H_calc_fpp),
            'n_max_fpp': float(n_max_fpp),
            'n_max_spp': float(n_max_spp),
            'rs_floor_fpp': float(rs_floor_fpp),
            'rs_floor_spp': float(rs_floor_spp)
        }


    @staticmethod
    def is_Rs_below_floor() -> bool:
        current_rs = Efficiency._get_safe_last('all_resolutions_final', 0.0)
        current_is_spp = Efficiency._get_safe_last('all_coreshells', False)
        
        floors = Efficiency.calculate_rs_floor()
        active_floor = floors['rs_floor_spp'] if not current_is_spp else floors['rs_floor_fpp']
        print(f'Rs Floor is {active_floor}')

        return sns(
            evaluate = current_rs < active_floor,
            floor = active_floor
        )

    @staticmethod
    def diagnose_flow_failure() -> str:
        current_bp = Efficiency._get_safe_last('all_backpressures', 1000.0)
        current_flow = Efficiency._get_safe_last('all_flow_rates', 1.0)
        bpspec = st.session_state.get('bpspec', 5000.0)

        bp_at_max = current_bp * (2.0 / current_flow) if current_flow > 0 else current_bp
        if bp_at_max > bpspec:
            return f"Flow rate increase is constrained by pressure limit (at 2.0 mL/min, BP reaches {bp_at_max:.1f} psi vs {bpspec:.1f} psi spec)."

        return "Flow rate adjustments alone cannot yield sufficient theoretical plates ($N$) to achieve required resolution."
        
    @staticmethod
    def handle_efficiency(ignore_parameters: list = []) -> sns:
        """Evaluates optimization options and registers candidate state for UI rendering in main()."""

        current_L = float(Efficiency._get_safe_last('all_column_lengths', 150.0))
        current_dp = float(Efficiency._get_safe_last('all_particle_sizes', 5.0))
        current_spp = bool(Efficiency._get_safe_last('all_coreshells', False))
        current_F = float(Efficiency._get_safe_last('all_flow_rates', 1.0))

        current_bp = float(Efficiency._get_safe_last('all_backpressures', 1000.0))
        current_tR = float(Efficiency._get_safe_last('all_run_times', 10.0))
        current_rs = float(Efficiency._get_safe_last('all_resolutions_final', 1.0))
        current_N = float(Efficiency._get_safe_last('plates', 10000.0))

        bpspec = float(st.session_state.get('bpspec', 4000.0))
        runspec = float(st.session_state.get('runspec', 15.0))
        res_spec = float(st.session_state.get('res_spec', 1.4))

        ret_values = st.session_state.get('all_retention_times', [])
        if not ret_values or len(ret_values[-1]) == 0:
            st.warning("No retention time data available for scaling.")
            return None

        all_tzeros = [tx[0] for tx in ret_values]
        tzero = float(np.mean(all_tzeros))

        if current_rs >= res_spec and current_tR <= runspec:
            st.info("Method is fully within specification. No adjustments required!")
            return None

        u1 = current_L / (tzero * 60.0)
        spp_factor1 = 1.5 if current_spp else 1.0
        hetp1 = (1.5 * current_dp + (3.0 / u1) + 0.15 * (current_dp ** 2) * u1) / spp_factor1

        L_grid = [current_L] if 'length' in ignore_parameters else st.session_state.get('available_column_lengths', [50.0, 75.0, 100.0, 150.0, 250.0])
        dp_grid = [current_dp] if 'particle_size' in ignore_parameters else st.session_state.get('available_particle_sizes', [1.7, 1.8, 2.5, 3.5, 5.0])
        spp_grid = [current_spp] if 'spp' in ignore_parameters else st.session_state.get('available_spp_states', [False, True])
        F_grid = [current_F] if 'flow_rate' in ignore_parameters else [float(round(f, 2)) for f in np.arange(0.30, 2.05, 0.05)]

        def evaluate_candidate(L2, dp2, spp2, F2):
            spp_factor2 = 1.5 if spp2 else 1.0
            u2 = u1 * (F2 / current_F)
            if u2 <= 0:
                return None

            hetp2 = (1.5 * dp2 + (3.0 / u2) + 0.15 * (dp2 ** 2) * u2) / spp_factor2
            gamma = (L2 / current_L) * (hetp1 / hetp2)

            predicted_rs = current_rs * np.sqrt(gamma)
            predicted_N = current_N * gamma

            #spp_bp_mult = 0.85 if (not current_spp and spp2) else (1.15 if (current_spp and not spp2) else 1.0)
            spp_bp_mult = 1.0
            predicted_bp = current_bp * (F2 / current_F) * (L2 / current_L) * ((current_dp / dp2) ** 2) * spp_bp_mult
            predicted_tR = current_tR * (current_F / F2) * (L2 / current_L)

            if predicted_bp <= bpspec and predicted_tR <= runspec and predicted_rs >= res_spec:
                changes = []
                equip_type = set()

                if not np.isclose(F2, current_F):
                    changes.append(f"Flow Rate: {current_F:.2f} -> {F2:.2f} mL/min")
                    equip_type.add("pump")
                if not np.isclose(L2, current_L):
                    changes.append(f"Column Length: {current_L:.0f} -> {L2:.0f} mm")
                    equip_type.add("column")
                if not np.isclose(dp2, current_dp):
                    changes.append(f"Particle Size: {current_dp:.1f} -> {dp2:.1f} µm")
                    equip_type.add("column")
                if spp2 != current_spp:
                    changes.append(f"Particle Type: {'Superficially Porous (SPP)' if spp2 else 'Fully Porous (FPP)'}")
                    equip_type.add("column")

                tier = len(changes)
                if tier == 0:
                    return None

                solvent_used = F2 * predicted_tR
                hardware_cost = 0.0
                if "column" in equip_type:
                    hardware_cost += 50.0 if spp2 == current_spp else 100.0

                if "pump" in equip_type and "column" in equip_type:
                    equipment_str = "pump and column"
                elif "pump" in equip_type:
                    equipment_str = "pump"
                else:
                    equipment_str = "column"

                justification = (
                    f"Consumes {solvent_used:.1f} mL solvent per run ({predicted_tR:.2f} min analysis time). "
                    f"Hardware investment rating (0-100): ${hardware_cost:.0f}."
                )

                return {
                    'L': L2, 'dp': dp2, 'is_spp': spp2, 'flow_rate': F2,
                    'bp': predicted_bp, 'tR': predicted_tR, 'plates': predicted_N,
                    'predicted_rs': predicted_rs, 'tier': tier, 'changes': changes,
                    'equipment': equipment_str, 'solvent_used': solvent_used,
                    'hardware_cost': hardware_cost, 'justification': justification
                }
            return None

        # Gather Tier 1 candidates first
        tier_1_candidates = []
        for fn, param_name, attr_name in [
            (Efficiency.flow_rate, 'flow_rate', 'flow_rate'),
            (Efficiency.column_length, 'length', 'length'),
            (Efficiency.particle_size, 'particle_size', 'size'),
            (Efficiency.spp, 'spp', 'is_spp')
        ]:
            if param_name not in ignore_parameters:
                res = fn()
                if res is not None:
                    args = [current_L, current_dp, current_spp, current_F]
                    if param_name == 'flow_rate': args[3] = res['flow_rate']
                    elif param_name == 'length': args[0] = res['length']
                    elif param_name == 'particle_size': args[1] = res['size']
                    elif param_name == 'spp': args[2] = res['is_spp']
                    
                    cand = evaluate_candidate(*args)
                    if cand:
                        tier_1_candidates.append(cand)

        selected_candidates = tier_1_candidates

        # Combinatorial search for higher tiers if Tier 1 yields no results
        if not selected_candidates:
            candidate_buckets = {2: [], 3: [], 4: []}
            for L2 in L_grid:
                for dp2 in dp_grid:
                    for spp2 in spp_grid:
                        for F2 in F_grid:
                            cand = evaluate_candidate(L2, dp2, spp2, F2)
                            if cand and cand['tier'] in candidate_buckets:
                                candidate_buckets[cand['tier']].append(cand)

            for tier_level in [2, 3, 4]:
                if candidate_buckets[tier_level]:
                    selected_candidates = candidate_buckets[tier_level]
                    break

        if selected_candidates:
            # 1. Base sort by efficiency metrics (Lowest solvent, minimum runtime)
            selected_candidates.sort(key=lambda x: (x['solvent_used'], x['tR']))

            # 2. Get active ligand (case-insensitive & whitespace trimmed)
            current_ligand = str(Efficiency._get_safe_last('all_ligands', 'C18')).strip().upper()
            inventory_columns = st.session_state.get('available_columns', [])

            # Helper to check if candidate matches an in-stock column configuration
            def is_in_possession(cand):
                for col in inventory_columns:
                    col_ligand = str(col.get('ligand', 'C18')).strip().upper()
                    # Approximate/flexible ligand matching (e.g. "C18" in "AQ-C18" or exact substring)
                    ligand_match = (col_ligand in current_ligand) or (current_ligand in col_ligand)
                    
                    if (
                        np.isclose(col['length'], cand['L']) and
                        np.isclose(col['dp'], cand['dp']) and
                        bool(col['is_spp']) == bool(cand['is_spp']) and
                        ligand_match
                    ):
                        return True
                return False

            # Tag candidates and partition into possession vs. non-possession
            candidates_in_possession = []
            candidates_not_in_possession = []

            for cand in selected_candidates:
                in_stock = is_in_possession(cand)
                cand['is_in_stock'] = in_stock
                cand['hardware_cost'] = 0.0 if in_stock else 500.0
                
                # Append status tag to justification text
                tag = " [IN STOCK]" if in_stock else " [PURCHASE NEEDED]"
                cand['justification'] = f"{cand['justification']} Stock Status:{tag}"

                if in_stock:
                    candidates_in_possession.append(cand)
                else:
                    candidates_not_in_possession.append(cand)

            # 3. Stack in-possession columns directly at the top of the pile
            st.session_state.efficiency_candidates = candidates_in_possession + candidates_not_in_possession
            st.session_state.show_efficiency_ui = True
            return None

        # PRESERVED: Fallback to Selectivity when efficiency cannot resolve deficit
        st.warning("Efficiency adjustments alone cannot resolve method deficits.")
        handle_selectivity()
        return None


    @staticmethod
    def commit_efficiency_selection(selected_candidate: dict):
        """Updates session state controlled parameters and renders the banner."""
        static_keys = [
            'all_solvents', 'all_ligands', 'all_bead_types',
            'all_pore_sizes', 'all_coreshells', 'all_carbon_loads',
            'all_column_lengths', 'all_internal_diameters', 'all_particle_sizes',
            'all_flow_rates', 'all_temperatures', 'all_pct_bs'
        ]

        def safe_append(key, val):
            if key not in st.session_state or not isinstance(st.session_state[key], list):
                st.session_state[key] = []
            st.session_state[key].append(val)

        modified_parameters = []

        # Modify explicitly changed controlled keys
        safe_append('all_flow_rates', selected_candidate['flow_rate'])
        modified_parameters.append('all_flow_rates')

        safe_append('all_column_lengths', selected_candidate['L'])
        modified_parameters.append('all_column_lengths')

        safe_append('all_particle_sizes', selected_candidate['dp'])
        modified_parameters.append('all_particle_sizes')

        safe_append('all_coreshells', selected_candidate['is_spp'])
        modified_parameters.append('all_coreshells')

        # Replicate unmodified static controlled keys
        for key in static_keys:
            if key not in modified_parameters:
                if key in st.session_state and isinstance(st.session_state[key], list) and st.session_state[key]:
                    st.session_state[key].append(st.session_state[key][-1])

        # Render diagnostic banner
        changes_str = "; ".join(selected_candidate['changes'])
        st.info(
            f"**Next Run Conditions (Cycle: Efficiency)**\n\n"
            f"- **Equipment Required:** {selected_candidate['equipment'].capitalize()}\n"
            f"- **Adjustments:** {changes_str}\n\n"
            f"**Projected Target Metrics:**\n"
            f"- Resolution ($R_s$): **{selected_candidate['predicted_rs']:.2f}**\n"
            f"- Run Time ($t_R$): **{selected_candidate['tR']:.2f} min**\n"
            f"- Backpressure ($P$): **{selected_candidate['bp']:.1f} psi**\n"
            f"- Solvent Consumption: **{selected_candidate['solvent_used']:.1f} mL**"
        )
        st.success("Successfully committed efficiency conditions to session state!")

        # Close UI block and clean candidate state
        st.session_state.show_efficiency_ui = False
        st.session_state.efficiency_candidates = []


    
   




def extract_measured_temp(df: pd.DataFrame) -> float:
    """
    Searches column 0 for 'oven temp', moves 9 rows down and 1 column right,
    and returns the measured temperature as a float.
    """
    try:
        # Search for 'oven temp' in the dataframe (case-insensitive)
        for r_idx in range(len(df)):
            cell_val = str(df.iloc[r_idx, 0]).lower()
            if 'oven temp' in cell_val:
                # Target: 9 rows down, 1 column to the right
                target_row = r_idx + 9
                target_col = 1
                
                if target_row < len(df) and target_col < len(df.columns):
                    val = df.iloc[target_row, target_col]
                    measured_temp = float(val)
                    return measured_temp
    except Exception as e:
        print(f"[DEBUG extract_measured_temp] Could not parse oven temp: {e}")
    
    # Fallback default if not found
    return 25.0

def calculate_temp_adjusted_minb(predicted_min_b: float, measured_temp: float, new_temp: float = 40.0) -> float:
    """
    Calculates a conservative temperature-adjusted min %B using:
    tR(new) = tR(old) * (1 - 0.01 * delta_T)
    Applies math.ceil() to the output to prevent runtime overshoot.
    """
    delta_t = new_temp - measured_temp
    if delta_t <= 0:
        return float(ceil(predicted_min_b))
    
    # Conservative factor f = 0.01 per degree C
    f = 0.01
    retention_scale_factor = 1.0 - (f * delta_t)
    
    # Temperature reduces retention, enabling a lower %B requirement for equal elution speed
    # Scaling factor applied to predicted %B threshold
    adjusted_b = predicted_min_b * retention_scale_factor
    
    # Conservative ceiling to guarantee zero runtime overshoot
    return float(ceil(adjusted_b))

def normalize_temp_value(temp_input) -> float:
    """
    Strips non-numeric characters ('T', '°C', spaces) using built-in string filtering.
    Converts 'T40', '40°C', or ' 40.0 ' to a float float(40.0).
    """
    if temp_input is None:
        return 0.0

    raw_str = str(temp_input)
    
    # Keep only digits and decimal point
    cleaned_str = "".join(char for char in raw_str if char.isdigit() or char == ".")
    
    try:
        return float(cleaned_str)
    except ValueError:
        return 0.0

def render_min_b_decision_ui(predicted_min_b: float, measured_temp: float, target_temp: float = 40.0) -> float:
    """
    Renders single-selection pills comparing standard predicted Min %B vs. temperature-adjusted Min %B.
    Uses column t0 from session state for accurate runtime estimates.
    """
    st.subheader(f"Temperature-Adjusted %B Strategy ({target_temp:.0f} °C)")
    
    adj_min_b = calculate_temp_adjusted_minb(predicted_min_b, measured_temp, new_temp=target_temp)
    
    # Extract actual t0 from session state if available, else default to 1.2 min
    t0 = st.session_state.get('all_retention_times', [[1.2]])[0][0] if st.session_state.get('all_retention_times') else 1.2
    
    # Chromatographically realistic runtime estimations
    est_runtime_std = round(t0 * (1.0 + (100.0 / max(predicted_min_b, 10.0))), 1)
    scale_factor = 0.85 if target_temp == 40.0 else 0.70
    est_runtime_adj = round(t0 * (1.0 + (100.0 / max(adj_min_b, 10.0))) * scale_factor, 1)

    opt_std_label = f"Continue with predicted Min %B: {predicted_min_b:.1f}% (Est. Runtime: ~{est_runtime_std} min)"
    opt_adj_label = f"Use temperature-adjusted Min %B: {adj_min_b:.1f}% (Est. Runtime: ~{est_runtime_adj} min)"

    options_map = {
        opt_std_label: float(predicted_min_b),
        opt_adj_label: float(adj_min_b)
    }

    pills_key = f"temp_opt_pills_{st.session_state.get('selectivity_progression', 0)}"

    # Default to adjusted option if not previously selected
    if pills_key not in st.session_state:
        st.session_state[pills_key] = opt_adj_label

    selected_label = st.pills(
        "Select target mobile phase composition for elevated temperature run:",
        options=list(options_map.keys()),
        selection_mode="single",
        key=pills_key
    )

    # Fallback to standard option if unselected
    if not selected_label:
        selected_label = opt_std_label

    chosen_b = options_map[selected_label]
    st.session_state.selected_min_b = chosen_b
    return chosen_b

def safe_min_b(tr):
    try:
        return min_b(tr)
    except Exception as e:
        st.info(f"ℹ️ Not enough valid data points for Min %B calculation ({e}). Falling back to current %B baseline.")
        if st.session_state.all_pct_bs:
            print(f"[DEBUG safe_min_b] Falling back to last known %B: {st.session_state.all_pct_bs[-1]}%")
            print(e)
            return float(st.session_state.all_pct_bs[-1])
        return 50.0

def safe_int_pct_b():
    try:
        return int_pct_b()
    except Exception as e:
        if st.session_state.all_pct_bs:
            print(f"[DEBUG safe_int_pct_b] Falling back to last known %B: {st.session_state.all_pct_bs[-1]}%")
            print(e)
            return float(st.session_state.all_pct_bs[-1])
        return 50.0

def on_override_segmented_control_change(control_key: str):
    """
    Callback function triggered the exact millisecond the user selects an option 
    in st.segmented_control. Guarantees immediate execution of handle_selectivity.
    """
    selected_val = st.session_state.get(control_key)
    if selected_val == "Switch to Selectivity Optimization":
        # Force execution of selectivity progression logic
        handle_selectivity(prev_b=False)


def evaluate_next_run_conditions(peakspec, runspec, res_spec, bpspec):
    """
    Evaluates chromatographic metrics of the latest run and displays actionable next run strategy.
    Guarantees proposed values (F, L, dp, SPP) are displayed and appended to session_state arrays.
    """
    required_state_keys = [
        'plates', 'heights', 'required_plates_history', 'all_coreshells',
        'all_backpressures', 'all_peak_counts', 'all_run_times',
        'all_resolutions_final', 'all_equilibrated', 'all_column_lengths',
        'all_particle_sizes', 'all_flow_rates', 'all_internal_diameters',
        'all_retention_times', 'all_pct_bs'
    ]
    for key in required_state_keys:
        if key not in st.session_state:
            st.session_state[key] = []

    if not st.session_state.all_backpressures:
        return

    latest_bp = st.session_state.all_backpressures[-1]
    latest_peaks = st.session_state.all_peak_counts[-1]
    latest_runtime = st.session_state.all_run_times[-1]
    latest_resolution = st.session_state.all_resolutions_final[-1]
    latest_equilibration = st.session_state.all_equilibrated[-1]
    latest_N = st.session_state.plates[-1] if st.session_state.plates else 1.0
    run_count = len(st.session_state.all_backpressures)
    
    static_keys = [
        'all_solvents', 'all_ligands', 'all_bead_types',
        'all_pore_sizes', 'all_coreshells', 'all_carbon_loads',
        'all_column_lengths', 'all_internal_diameters', 
        'all_particle_sizes', 'all_flow_rates', 'all_temperatures'
    ]
    
    if latest_equilibration:
        st.success('Backpressure is stable, column is equilibrated well.')

    def update_session_state_keys(new_length=None, new_ps=None, new_flow=None, is_spp=None):
        st.session_state.all_pct_bs.append(st.session_state.all_pct_bs[-1])
        for key in static_keys:
            val = st.session_state[key][-1]
            if key == 'all_column_lengths' and new_length is not None:
                val = new_length
            elif key == 'all_particle_sizes' and new_ps is not None:
                val = new_ps
            elif key == 'all_flow_rates' and new_flow is not None:
                val = new_flow
            elif key == 'all_coreshells' and is_spp is not None:
                val = 'Core-Shell' if is_spp else val
            st.session_state[key].append(val)

    def display_strategy_and_update(action_title, eval_result, controller_text="Column / Pump"):
        if eval_result is not None:
            if 'plates' in eval_result:
                pred_N = eval_result['plates']
            else:
                pred_N = latest_N
                if 'length' in eval_result:
                    pred_N *= (eval_result['length'] / st.session_state.all_column_lengths[-1])
                if 'size' in eval_result:
                    pred_N *= (st.session_state.all_particle_sizes[-1] / eval_result['size'])
                if eval_result.get('is_spp'):
                    pred_N *= 1.5

            pred_Rs = latest_resolution * ((pred_N / latest_N) ** 0.5)
            
            # Format proposed specifications explicitly
            proposed_details = []
            if 'flow_rate' in eval_result:
                proposed_details.append(f"$F$: {eval_result['flow_rate']:.2f} mL/min")
            if 'length' in eval_result:
                proposed_details.append(f"$L$: {eval_result['length']} mm")
            if 'size' in eval_result:
                proposed_details.append(f"$d_p$: {eval_result['size']} µm")
            if eval_result.get('is_spp'):
                proposed_details.append("Core-Shell (SPP)")

            param_str = f" ({', '.join(proposed_details)})" if proposed_details else ""

            st.info(
                f"**Next Run Strategy:** {action_title}{param_str}\n\n"
                f"**Predicted Next Conditions:** $t_R$ {eval_result['tR']:.1f} min, "
                f"BP {eval_result['bp']:.1f} psi, $R_s$ {pred_Rs:.2f}\n\n"
                f"*Controlled by:* {controller_text} | *Cycle:* Efficiency (N)"
            )
            update_session_state_keys(
                new_length=eval_result.get('length'),
                new_ps=eval_result.get('size'),
                new_flow=eval_result.get('flow_rate'),
                is_spp=eval_result.get('is_spp')
            )
        else:
            failure_reason = Efficiency.diagnose_flow_failure()
            
            st.warning(
                f"**Next Run Strategy:** {action_title}\n\n"
                f"**Physical Limitation:** {failure_reason}\n\n"
                f"*Controlled by:* {controller_text} | *Cycle:* Efficiency (N)"
            )
            update_session_state_keys()

            ctrl_key = f"override_ctrl_{run_count}"

            # Streamlit segmented control using on_change callback
            st.segmented_control(
                "Efficiency specs unreachable. Select action:",
                options=["Continue Current Physical Search", "Switch to Selectivity Optimization"],
                default="Continue Current Physical Search",
                key=ctrl_key,
                on_change=on_override_segmented_control_change,
                args=(ctrl_key,)
            )

    # --- 1. Backpressure Check ---
    if latest_bp > bpspec:
        st.caption(f"📊 **Run Metrics:** Backpressure ({latest_bp:.1f} psi) > Spec ({bpspec:.1f} psi) | Peaks: {latest_peaks}/{peakspec}")
        st.warning(f'Run number {run_count} exceeded backpressure limit!')

    # --- 2. Too Many Peaks ---
    elif latest_peaks > peakspec:
        st.caption(f"📊 **Run Metrics:** Peaks ({latest_peaks}) > Spec ({peakspec}) | $t_R$: {latest_runtime:.1f} min | $R_s$: {latest_resolution:.2f}")
        st.warning(f'Run number {run_count} has more peaks than expected. Investigate for contamination.')

    # --- 3. Fewer Peaks Than Spec ---
    elif latest_peaks < peakspec:
        if latest_runtime < (0.5 * runspec):
            st.caption(f"📊 **Run Metrics:** Peaks ({latest_peaks}) < Spec ({peakspec}) | $t_R$ ({latest_runtime:.1f} min) << Spec ({runspec} min)")
            current_pct_b = st.session_state.all_pct_bs[-1]
            new_pct_b = current_pct_b - 10.0
            
            st.info(
                f"**Next Run Strategy:** Decrease %B by 10% (from {current_pct_b}% to {new_pct_b}%)\n\n"
                f"*Controlled by:* Pump | *Cycle:* Retention"
            )
            st.session_state.all_pct_bs.append(new_pct_b)
            for key in static_keys:
                st.session_state[key].append(st.session_state[key][-1])

        elif latest_runtime < runspec:
            if not st.session_state.get('minbdone', False):
                st.caption(f"📊 **Run Metrics:** Peaks ({latest_peaks}) < Spec ({peakspec}) | $t_R$ ({latest_runtime:.1f} min) < Spec ({runspec} min)")
                c1 = safe_min_b(runspec)
                c2 = safe_int_pct_b()
                st.session_state.choice1 = c1
                st.session_state.choice2 = c2
                st.info(
                    f"**Next Run Strategy Options:**\n"
                    f"- Option A (Min %B): Set %B to {c1}%\n"
                    f"- Option B (Int %B): Set %B to {c2}%\n\n"
                    f"*Controlled by:* Pump | *Cycle:* Retention"
                )
                st.session_state.show_decision_radio = True
                st.session_state.minbdone = True
            else:
                st.caption(f"📊 **Run Metrics:** Peaks ({latest_peaks}) < Spec ({peakspec}) | min%B already executed -> Triggering Selectivity.")
                handle_selectivity(prev_b=False)

        elif latest_runtime > runspec:
            st.caption(f"📊 **Run Metrics:** Peaks ({latest_peaks}) < Spec ({peakspec}) | $t_R$ ({latest_runtime:.1f} min) > Spec ({runspec} min)")
            handle_selectivity(prev_b=False)

    # --- 4. Target Peak Count Met (Matrix Evaluation) ---
    # --- 4. Target Peak Count Met (Matrix Evaluation) ---
    elif latest_peaks == peakspec:
        tR_far_below = latest_runtime < (0.5 * runspec)
        tR_below = (latest_runtime < runspec) and not tR_far_below
        tR_above = latest_runtime > runspec
        pass_resolution = latest_resolution >= res_spec
        rs_floor_ns = Efficiency.is_Rs_below_floor()
        rs_below_floor = rs_floor_ns.evaluate
        rs_floor = rs_floor_ns.floor

        if pass_resolution:
            run_metrics = (
                f"📊 **Run Metrics:** Peaks ({latest_peaks}) == Spec ({peakspec}) "
                f"| $R_s$ ({latest_resolution:.2f}) >= Spec ({res_spec}) "
                f"| Runtime ({latest_runtime:.2f}) "
            )
            if tR_far_below or tR_below:
                if tR_far_below:
                    run_metrics += '<< Spec'
                else:
                    run_metrics += '< Spec'
                st.caption(run_metrics)
                st.success(
                    "🎉 **Method MET!** Target peak count, resolution, and analysis time specification are all achieved.\n\n"
                    "*(Optionally increase Flow Rate ($F$) to shorten run time further. Controlled by: Pump | Cycle: Efficiency (N))*"
                )
                update_session_state_keys()

            elif tR_above:
                new_f = Efficiency.flow_rate()
                excerpt = ''
                if new_f is not None:
                    excerpt = (
                        f" Increasing flow rate ($F$) directly reduces linear linear velocity residence time via $t_0 = V_m / F$, "
                        f"shortening analysis time without altering chromatographic thermodynamics.\n"
                        f"Expected next run conditions: $F$ = {new_f['flow_rate']:.2f} mL/min.\n"
                        f"Predicted Metrics -> $R_s$: {new_f['rs']:.2f} | $N$: {new_f['plates']:.0f} | "
                        f"Runtime: {new_f['tR']:.2f} min | Pressure: {new_f['bp']:.0f} psi\n"
                        f"Cycle: Efficiency | Equipment: Pump"
                    )
                else:
                    reason = Efficiency.diagnose_flow_failure() if hasattr(Efficiency, 'diagnose_flow_failure') else "flow rate limits alone cannot meet specifications"
                    excerpt = f" However, {reason[0:1].lower()+reason[1:]} Tool has automatically escalated to candidate combinations."
                st.info(f"Prescribed Strategy: Increase Flow Rate ($F$) to shorten run time.{excerpt}")
                run_metrics += '> Spec'
                st.caption(run_metrics)
                if 'However' in excerpt:
                    Efficiency.handle_efficiency()

        else:
            # =========================================================================
            # CASE A: RUNTIME FAR BELOW SPEC (tR << Spec) & RESOLUTION DEFICIT
            # =========================================================================
            if tR_far_below:
                if rs_below_floor:
                    # The requirements indeed say to Increase L even though Rs below floor. Definitely verify this.
                    st.caption(f"📊 **Run Metrics:** Peaks ({latest_peaks}) == Spec ({peakspec}) | $t_R$ << Spec | $R_s$ below floor")
                    res = Efficiency.column_length()
                    excerpt = ''
                    if res is not None:
                        excerpt = (
                            f" Severe resolution deficit ($R_s < {rs_floor:.2f}$) requires a major increase in plate count ($N \\propto L$). "
                            f"Since runtime is far below specification ($t_R < 0.5 \\times t_{{R,\\text{{spec}}}}$), increasing column length ($L$) provides "
                            f"the necessary efficiency gains while keeping $t_R$ safely within bounds.\n"
                            f"Predicted Metrics -> $L$: {res['length']:.0f} mm | $R_s$: {res['rs']:.2f} | $N$: {res['plates']:.0f} | "
                            f"Runtime: {res['tR']:.2f} min | Pressure: {res['bp']:.0f} psi\n"
                            f"Cycle: Efficiency | Equipment: Column"
                        )
                    else:
                        excerpt = " However, increasing column length alone is insufficient or exceeds system backpressure limits ($P \\propto L$). Tool has automatically escalated to candidate combinations."

                    st.info(f"Prescribed Strategy: Increase Column Length ($L$) to boost efficiency ($N$).{excerpt}")
                    if 'However' in excerpt:
                        Efficiency.handle_efficiency()

                else:
                    st.caption(f"📊 **Run Metrics:** Peaks ({latest_peaks}) == Spec ({peakspec}) | $t_R$ << Spec | Floor < $R_s$ < Spec")
                    candidates = [
                        Efficiency.column_length(),
                        Efficiency.flow_rate(),
                        Efficiency.particle_size(),
                        Efficiency.spp()
                    ]
                    valid = [c for c in candidates if c is not None]
                    best = min(valid, key=lambda x: x['tR']) if valid else None
                    excerpt = ''
                    if best is not None:
                        excerpt = (
                            f" Moderate resolution deficit with significant runtime budget available ($t_R < 0.5 \\times t_{{R,\\text{{spec}}}}$). "
                            f"Multiple single-parameter options exist ($L \\uparrow$, $F \\downarrow$, $d_p \\downarrow$, or Core-Shell SPP).\n"
                            f"Optimal single adjustment selected: {best.get('description', 'Efficiency Parameter Adjustment')}.\n"
                            f"Predicted Metrics -> $R_s$: {best['rs']:.2f} | $N$: {best['plates']:.0f} | "
                            f"Runtime: {best['tR']:.2f} min | Pressure: {best['bp']:.0f} psi\n"
                            f"Cycle: Efficiency | Equipment: {best.get('equipment', 'Column/Pump').capitalize()}"
                        )
                    else:
                        excerpt = " However, no single-parameter adjustment satisfies both $R_s$ and pressure limits simultaneously. Tool has automatically escalated to candidate combinations."

                    st.info(f"Prescribed Strategy: Increase $L$, decrease $F$, decrease $d_p$, or switch to SPP.{excerpt}")
                    if 'However' in excerpt:
                        Efficiency.handle_efficiency()

            # =========================================================================
            # CASE B: RUNTIME BELOW SPEC (tR < Spec) & RESOLUTION DEFICIT
            # =========================================================================
            elif tR_below:
                # VALIDATED! 
                if rs_below_floor:
                    st.caption(f"📊 **Run Metrics:** Peaks ({latest_peaks}) == Spec ({peakspec}) | $t_R$ < Spec | $R_s$ below floor -> Triggering Selectivity.")
                    st.warning(
                        f"⚠️ **Resolution is below the hard efficiency floor ($R_s < {rs_floor:.2f}$) with tight runtime remaining.**\n\n"
                        "Gaining sufficient resolution through efficiency alone requires $N \\propto R_s^2$ (a $4\\times$ plate increase for $2\\times$ $R_s$), "
                        "which would breach the runtime and/or backpressure specification. **Escalating directly to Selectivity (α) Cycle** to shift retention order."
                    )
                    handle_selectivity(prev_b=False)
                else:
                    st.caption(f"📊 **Run Metrics:** Peaks ({latest_peaks}) == Spec ({peakspec}) | $t_R$ < Spec | Floor < $R_s$ < Spec")
                    candidates = [
                        Efficiency.flow_rate(),
                        Efficiency.particle_size(),
                        Efficiency.spp()
                    ]
                    valid = [c for c in candidates if c is not None]
                    best = min(valid, key=lambda x: x['tR']) if valid else None
                    excerpt = ''
                    if best is not None:
                        excerpt = (
                            f" Moderate resolution deficit with modest runtime margin. Decreasing flow rate ($F$), reducing particle size ($d_p$), "
                            f"or switching to Core-Shell (SPP) technology sharpens efficiency without the severe runtime penalty of increasing column length.\n"
                            f"Optimal single adjustment selected: {best.get('description', 'Efficiency Parameter Adjustment')}.\n"
                            f"Predicted Metrics -> $R_s$: {best['rs']:.2f} | $N$: {best['plates']:.0f} | "
                            f"Runtime: {best['tR']:.2f} min | Pressure: {best['bp']:.0f} psi\n"
                            f"Cycle: Efficiency | Equipment: {best.get('equipment', 'Pump/Column').capitalize()}"
                        )
                    else:
                        excerpt = " However, single adjustments cannot bridge the resolution gap within the remaining runtime budget. Tool has automatically escalated to candidate combinations."

                    st.info(f"Prescribed Strategy: Decrease $F$, decrease $d_p$, or switch to SPP.{excerpt}")
                    if 'However' in excerpt:
                        Efficiency.handle_efficiency()

            # =========================================================================
            # CASE C: RUNTIME ABOVE SPEC (tR > Spec) & RESOLUTION DEFICIT
            # =========================================================================
            elif tR_above:
                if rs_below_floor:
                    st.caption(f"📊 **Run Metrics:** Peaks ({latest_peaks}) == Spec ({peakspec}) | $t_R$ > Spec | $R_s$ below floor -> Triggering Selectivity.")
                    st.warning(
                        f"⚠️ **Method exceeds runtime specification ($t_R > t_{{R,\\text{{spec}}}}$) AND resolution is below floor ($R_s < {rs_floor:.2f}$).**\n\n"
                        "Increasing column length or lowering flow rate to gain efficiency would exacerbate the runtime defect. "
                        "**Escalating directly to Selectivity (α) Cycle** to resolve the critical peak pair chemically."
                    )
                    handle_selectivity(prev_b=False)
                else:
                    st.caption(f"📊 **Run Metrics:** Peaks ({latest_peaks}) == Spec ({peakspec}) | $t_R$ > Spec | Floor < $R_s$ < Spec")
                    res_dp = Efficiency.particle_size()
                    res_f = Efficiency.flow_rate()
                    
                    excerpt = ''
                    if res_dp is not None:
                        excerpt = (
                            f" Decreasing particle size ($d_p$) significantly increases plate number ($N \\propto 1/d_p$) "
                            f"and lowers reduced plate height ($h$), enabling faster flow rates ($F \\uparrow$) without loss of efficiency.\n"
                            f"Predicted Metrics -> $d_p$: {res_dp['size']:.1f} µm | $R_s$: {res_dp['rs']:.2f} | $N$: {res_dp['plates']:.0f} | "
                            f"Runtime: {res_dp['tR']:.2f} min | Pressure: {res_dp['bp']:.0f} psi\n"
                            f"Cycle: Efficiency | Equipment: Column / Pump"
                        )
                    elif res_f is not None:
                        excerpt = (
                            f" Adjusting flow rate ($F$) re-optimizes linear velocity ($u$) along the Van Deemter curve, "
                            f"balancing plate count against runtime.\n"
                            f"Predicted Metrics -> $F$: {res_f['flow_rate']:.2f} mL/min | $R_s$: {res_f['rs']:.2f} | $N$: {res_f['plates']:.0f} | "
                            f"Runtime: {res_f['tR']:.2f} min | Pressure: {res_f['bp']:.0f} psi\n"
                            f"Cycle: Efficiency | Equipment: Pump"
                        )
                    else:
                        excerpt = " However, single-parameter changes cannot simultaneously shorten analysis time and recover resolution. Tool has automatically escalated to candidate combinations."

                    st.info(f"Prescribed Strategy: Decrease Particle Size ($d_p$), then adjust Flow Rate ($F$).{excerpt}")
                    if 'However' in excerpt:
                        Efficiency.handle_efficiency()

def generate_selectivity_plot():
    """Generates a simplified Plotly graph using stored slopes, intercepts, and R^2 values."""
    slopes = st.session_state.get('slopes', [])
    intercepts = st.session_state.get('intercepts', [])
    r_squareds = st.session_state.get('r_squareds', [])
    pct_bs = st.session_state.get('all_pct_bs', [])
    
    if not np.any(slopes) or not np.any(intercepts) or not np.any(pct_bs):
        return None
        
    fig = go.Figure()

    # Define x-axis range based on %B values
    b_min = max(0, min(pct_bs) - 5)
    b_max = min(100, max(pct_bs) + 5)
    b_range = np.linspace(b_min, b_max, 50)

    # Pad r_squareds if shorter than slopes list
    if len(r_squareds) < len(slopes):
        r_squareds.extend([0.0] * (len(slopes) - len(r_squareds)))

    # Iterate through slope, intercept, and R^2 triplets
    for idx, (slope, intercept, rsq) in enumerate(zip(slopes, intercepts, r_squareds)):
        # Convert values to scalar floats to prevent NumPy format errors
        slope_val = float(np.asarray(slope).squeeze())
        intercept_val = float(np.asarray(intercept).squeeze())
        rsq_val = float(np.asarray(rsq).squeeze())

        y_line = slope_val * b_range + intercept_val
        
        # Generate random hex color
        random_color = f"#{random.randint(0, 0xFFFFFF):06x}"
        
        fig.add_trace(go.Scatter(
            x=b_range,
            y=y_line,
            mode='lines',
            name=f'Peak {idx + 1} (R² = {rsq_val:.6f})',
            line=dict(color=random_color, width=2)
        ))

    fig.update_layout(
        title="Selectivity Plot: log10(k) vs % Organic Solvent (%B)",
        xaxis_title="% Organic Solvent (%B)",
        yaxis_title="log10(k)",
        template="plotly_white",
        height=450,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def determine_selectivity_case(actual_solvent, actual_temp, initial_solvent):
    """
    Determines the exact selectivity progression case based on physical run history,
    using substring matching for robust temperature parsing (e.g. '40.0', 'T40', '40°C').
    """
    sol_curr = str(actual_solvent).strip().upper()
    sol_init = str(initial_solvent).strip().upper()
    temp_str = str(actual_temp).strip().lower()
    
    # Substring detection for temperature values
    is_40 = "40" in temp_str
    is_60 = "60" in temp_str
    
    is_initial_solvent = (sol_curr == sol_init)
    
    # Check historical run temperatures for past temperature steps
    past_temps = [str(t).strip().lower() for t in st.session_state.get('all_temperatures', [])]
    had_60 = any("60" in t for t in past_temps)
    had_40 = any("40" in t for t in past_temps)
    
    if is_initial_solvent:
        if is_60 or had_60:
            return 2  # At 60 °C -> handle_selectivity does +1 -> Case 3 (Swap to MeOH at amb)
        elif is_40 or had_40:
            return 1  # At 40 °C -> handle_selectivity does +1 -> Case 2 (Increase to 60 °C)
        else:
            return 0  # At amb -> handle_selectivity does +1 -> Case 1 (Increase to 40 °C)
    else:
        if is_60:
            return 5  # Swapped solvent at 60 °C
        elif is_40:
            return 4  # Swapped solvent at 40 °C
        else:
            return 3  # Swapped solvent at amb -> handle_selectivity does +1 -> Case 4 (Swapped solvent at 40 °C)

def safe_read(uploaded_file):
    # 1. Detect encoding
    raw_bytes = uploaded_file.getvalue()
    detected = chardet.detect(raw_bytes)
    encoding = detected['encoding'] if detected['encoding'] else 'ascii'
    
    # 2. Decode text safely
    text = raw_bytes.decode(encoding, errors='ignore')
    
    # 3. Pre-pad every line with 16 trailing commas so every row has AT LEAST 16 columns
    padded_lines = [line + ',' * 16 for line in text.splitlines()]
    padded_text = '\n'.join(padded_lines)
    
    # 4. Read padded string into Pandas, slicing strictly columns 0 through 15 (A through P)
    df = pd.read_csv(
        io.StringIO(padded_text),
        header=None,
        usecols=list(range(16)),  # Grabs only indices 0..15
        engine='python'
    )
    
    return df

def retention_time(df, peaks):
    """
    Locates the 'Peak Table' anchor in the first column of the DataFrame and extracts
    the retention times for a specified number of peaks starting 3 rows below the anchor.
    
    Parameters:
        df (pd.DataFrame): The chromatogram DataFrame.
        peaks (int): The total number of peaks to extract.
        
    Returns:
        dict: A dictionary containing the list of retention times and total runtime.
    """
    # 1. Access the first column
    first_col = df.iloc[:, 0]
    target_string = 'Peak Table'
    
    # 2. Perform case-insensitive search for the target header
    matches = first_col.astype(str).str.contains(target_string, case=False, na=False)
    
    # 3. Guard clause: Ensure target string was actually found in the column
    if not matches.any():
        raise ValueError(f"'{target_string}' header was not found in the first column of the CSV.")
    
    # 4. Get the exact integer row index position using .values.argmax()
    row_of_match = matches.values.argmax()
    
    # 5. Define target row (3 rows below anchor) and target column (2nd column / index 1)
    start_row = row_of_match + 3
    required_column = 1
    
    # 6. Extract retention times for each requested peak
    retention_times = []
    for peak in range(peaks):
        val = df.iloc[start_row + peak, required_column]
        retention_times.append(float(val))
    
    # 7. Construct output dictionary (runtime is set to the last peak's retention time)
    output = {
        "retention_times": retention_times,
        "runtime": retention_times[-1] if retention_times else None
    }
    
    return output

def peak_count(df):
    """
    Locates '# of Peaks' in the first column and returns the count from Column B.
    """
    first_col = df.iloc[:, 0]
    target_string = '# of Peaks'
    
    matches = first_col.astype(str).str.contains(target_string, case=False, na=False)
    
    if not matches.any():
        raise ValueError(f"'{target_string}' header was not found in the first column of the CSV.")
    
    row_of_match = matches.values.argmax()
    
    # Cast to int to ensure numeric safety when passed into range()
    raw_val = df.iloc[row_of_match, 1]
    return int(raw_val)

def find_resolution(df, peaks):
    colnum = 15  # Column P
    res_col = df.iloc[:, colnum]
    target_string = 'Resolution'
    
    matches = res_col.astype(str).str.contains(target_string, case=False, na=False)
    
    if not matches.any():
        raise ValueError(f"'{target_string}' header was not found in column {colnum} of the CSV.")
    
    # Starts 1 row below the 'Resolution' header
    start_row = matches.values.argmax() + 1
    resolutions = []
    
    for peak in range(peaks):
        val = df.iloc[start_row + peak, colnum]
        
        # Safely parse numeric values (handles strings like '-', 'N/A', or NaNs)
        try:
            parsed_val = float(val)
        except (ValueError, TypeError):
            parsed_val = 0.0
            
        # Ignore zero or negative resolution values completely
        if parsed_val > 0.0:
            resolutions.append(parsed_val)
    
    # Calculate min resolution solely from non-zero values
    min_res = float(min(resolutions, default=0.0))
    
    output = {
        "resolutions": resolutions,
        "min_res": min_res,
        "min_res_no_zero": min_res
    }
    
    return output

def find_backpressure(df):
    first_col = df.iloc[:, 0].astype(str)
    
    # 1. Define individual boolean masks
    has_pump = first_col.str.contains('pump', case=False, na=False)
    has_pressure = first_col.str.contains('pressure', case=False, na=False)
    has_no_degas = ~first_col.str.contains('degas', case=False, na=False)
    
    # 2. Combine all conditions
    matches = has_pump & has_pressure & has_no_degas
    
    # 3. Guard check
    if not matches.any():
        raise ValueError("No cell in the first column matched criteria ('pump', 'pressure', no 'degas').")
        
    row_of_match = matches.values.argmax()
    required_column = 1  # Second Column
    
    # 4. Safely parse multiplier to float
    multiplier_row = row_of_match + 6
    try:
        multiplier = float(df.iloc[multiplier_row, required_column])
    except (ValueError, TypeError):
        multiplier = 1.0  # Fallback if cell is empty or non-numeric
    
    start_row = row_of_match + 8
    pressures = []
    
    for idx in range(100):
        target_row = start_row + idx
        
        # Stop if we reach the end of the DataFrame
        if target_row >= len(df):
            break
            
        val = df.iloc[target_row, required_column]
        
        # Convert value to float dynamically instead of strict type checking
        try:
            numeric_val = float(val)
            pressures.append(numeric_val)
        except (ValueError, TypeError):
            # Non-numeric cell encountered (e.g., end of table section)
            break
            
    if not pressures:
        raise ValueError("No numeric pressure values were found in the specified table range.")

    # 5. Split and compute stability metrics
    midpoint = len(pressures) // 2
    first_half = np.mean(pressures[:midpoint])
    second_half = np.mean(pressures[midpoint:])
    
    equilibrated = True
    if abs(first_half - second_half) >= (0.05 * second_half):
        equilibrated = False
        
    avg_pressure = float(np.mean([first_half, second_half])) * multiplier
    
    output = {
        'backpressure': avg_pressure,
        'equilibrated': equilibrated
    }
    
    return output

def min_b(tr):
    # Make copies so popping elements doesn't delete session state data
    runtimes = list(st.session_state.all_run_times)
    pct_bs = list(st.session_state.all_pct_bs)
    ret_values = st.session_state.all_retention_times
    all_tzeros = [tx[0] for tx in ret_values]
    tzero = np.mean(all_tzeros)
    #tzero = runtimes[0]
    
    # Calculate k and log10(k)
    k_values = [(runtime - tzero) / tzero for runtime in runtimes]
    
    # Safeguard against k <= 0 before taking log10
    log_ks = [np.log10(k) if k > 0 else -3.0 for k in k_values]
    
    slope1 = 0.0
    intercept1 = 0.0
    # Iterate over the length of the list, not the list itself
    valid_pct_bs = pct_bs
    for _ in range(len(valid_pct_bs)):
        if len(valid_pct_bs) < 2:
            break
        tzero = np.mean(all_tzeros)
        try:
            curve = linregress(valid_pct_bs, log_ks)
        except:
            raise ValueError(f'Valid: {str(valid_pct_bs)}. Log_ks: {str(log_ks)}. Runtimes: {str(runtimes)}. All pct_bs: {str(pct_bs)}. tzero: {tzero}')
        r_squared = round(curve.rvalue ** 2, 3)
        
        # Trim non-linear points if R² < 0.995 and enough data points remain
        if r_squared < 0.995 and len(pct_bs) > 4:
            log_ks.pop(0)
            valid_pct_bs.pop(0)
            all_tzeros.pop(0)
        else:
            slope1 = curve.slope
            intercept1 = curve.intercept
            break
            
    if slope1 == 0.0:
        raise ValueError("Linear regression failed: slope is zero.")
        
    # Correct calculation: required_logk needs np.log10
    k_target = (tr - tzero) / tzero
    if k_target <= 0:
        raise ValueError("Target retention time (tr) must be greater than t0.")
        
    required_logk = np.log10(k_target)
    
    # %B = (log10(k) - intercept) / slope
    min_pct_b = (required_logk - intercept1) / slope1
    
    print(f'Slope: {slope1}. Int: {intercept1}. RSQ: {r_squared}')
    st.session_state.tr_slope = slope1
    st.session_state.tr_intercept = intercept1
    st.session_state.tr_tzero = tzero
    return float(ceil(min_pct_b))



def int_pct_b():
    """
    Calculates the optimal Intermediate %B by fitting linear log10(k) vs %B models,
    refining peak alignment via Hungarian optimization to eliminate crossover errors,
    and maximizing worst-case peak separation strictly WITHIN the interior local window 
    of the last two screening runs (excluding boundary endpoints like 40% or 50%).
    """
    ret_times = list(st.session_state.get('all_retention_times', []))
    pct_bs = list(st.session_state.get('all_pct_bs', []))
    
    if not ret_times or len(ret_times) < 2:
        raise ValueError("At least two runs are required to perform linear regression.")
    
    # Calculate exact number of analyte peaks (excluding t0 at index 0)
    peak_counts = [len(sublist) - 1 for sublist in ret_times if len(sublist) > 1]
    if not peak_counts:
        return float(ceil(pct_bs[-1]))
        
    max_peaks = max(peak_counts)
    if max_peaks < 2:
        return float(ceil(pct_bs[-1]))

    # --------------------------------------------------------------------------
    # STEP 1: INITIAL DATA MATRIX CONSTRUCTION (Shape: num_runs x max_peaks)
    # --------------------------------------------------------------------------
    num_runs = len(ret_times)
    k_matrix = np.full((num_runs, max_peaks), np.nan, dtype=float)
    
    for run_idx in range(num_runs):
        if run_idx < len(pct_bs):
            run_ret_times = ret_times[run_idx]
            if len(run_ret_times) > 1:
                t0 = run_ret_times[0]
                # Populate analyte peaks 1..N into matrix columns 0..(N-1)
                for peak_idx in range(1, len(run_ret_times)):
                    if (peak_idx - 1) < max_peaks:
                        tR = run_ret_times[peak_idx]
                        if tR > t0 and t0 > 0:
                            k = (tR - t0) / t0
                            if k > 0:
                                k_matrix[run_idx, peak_idx - 1] = np.log10(k)

    def fit_models(data_mat):
        models = []
        n_cols = data_mat.shape[1]
        for p_idx in range(n_cols):
            y_vals = data_mat[:, p_idx]
            x_vals = np.array(pct_bs[:num_runs])
            valid_mask = ~np.isnan(y_vals)
            
            if np.sum(valid_mask) >= 2 and len(set(x_vals[valid_mask])) > 1:
                res = linregress(x_vals[valid_mask], y_vals[valid_mask])
                r2 = float(res.rvalue ** 2) if not np.isnan(res.rvalue) else 0.0
                models.append((float(res.slope), float(res.intercept), r2))
            else:
                models.append((0.0, 0.0, 0.0))
        return models

    current_models = fit_models(k_matrix)
    current_avg_r2 = np.mean([m[2] for m in current_models])
    
    # --------------------------------------------------------------------------
    # STEP 2: ITERATIVE HUNGARIAN OPTIMIZATION (Bipartite Peak Tracking)
    # --------------------------------------------------------------------------
    best_k_matrix = k_matrix.copy()
    best_models = current_models
    best_avg_r2 = current_avg_r2

    max_iterations = 5
    for iteration in range(max_iterations):
        candidate_k_matrix = np.full_like(best_k_matrix, np.nan)
        
        for run_idx in range(num_runs):
            run_pct_b = pct_bs[run_idx]
            actual_log_ks = best_k_matrix[run_idx, :]
            valid_indices = np.where(~np.isnan(actual_log_ks))[0]
            
            if len(valid_indices) == 0:
                continue
                
            obs_vals = actual_log_ks[valid_indices]
            n_obs = len(obs_vals)
            
            predicted_logs = np.array([m[0] * run_pct_b + m[1] for m in best_models])
            
            cost_matrix = np.zeros((n_obs, max_peaks), dtype=float)
            for i, obs in enumerate(obs_vals):
                for j, pred in enumerate(predicted_logs):
                    cost_matrix[i, j] = (obs - pred) ** 2

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            for r_idx, c_idx in zip(row_ind, col_ind):
                candidate_k_matrix[run_idx, c_idx] = obs_vals[r_idx]

        candidate_models = fit_models(candidate_k_matrix)
        candidate_avg_r2 = np.mean([m[2] for m in candidate_models])
        
        no_severe_degradation = all(
            cand[2] >= (prev[2] - 0.05) for cand, prev in zip(candidate_models, best_models)
        )
        
        if candidate_avg_r2 > best_avg_r2 and no_severe_degradation:
            best_avg_r2 = candidate_avg_r2
            best_models = candidate_models
            best_k_matrix = candidate_k_matrix
        else:
            break

    # --------------------------------------------------------------------------
    # STEP 3: PRESERVE OPTIMIZED CURVES & SESSION STATE
    # --------------------------------------------------------------------------
    optimised_curves = best_models
    
    st.session_state.slopes = [float(c[0]) for c in optimised_curves]
    st.session_state.intercepts = [float(c[1]) for c in optimised_curves]
    st.session_state.r_squareds = [round(float(c[2]), 6) for c in optimised_curves]

    if len(optimised_curves) < 2:
        return float(ceil(pct_bs[-1]))
        
    # --------------------------------------------------------------------------
    # STEP 4: SEPARATION OPTIMIZATION STRICTLY WITHIN INTERMEDIATE WINDOW
    # --------------------------------------------------------------------------
    # Filter for unique %B values that are multiples of 10 (e.g. 30, 40, 50)
    multiple_of_10_bs = []
    for b in reversed(pct_bs):
        val = float(b)
        # Check if value is a multiple of 10 (with floating-point tolerance)
        if abs(val % 10.0) < 1e-5 or abs(val % 10.0 - 10.0) < 1e-5:
            if val not in multiple_of_10_bs:
                multiple_of_10_bs.append(val)
            if len(multiple_of_10_bs) == 2:
                break

    if len(multiple_of_10_bs) == 2:
        b_min = min(multiple_of_10_bs)
        b_max = max(multiple_of_10_bs)
    else:
        # Fallback if two multiple-of-10 runs aren't present yet
        b_min = max(10.0, float(min(pct_bs)))
        b_max = min(95.0, float(max(pct_bs)))

    step = 0.5
    if b_min >= b_max:
        return float(ceil(pct_bs[-1]))

    # EXCLUDE ENDPOINTS (b_min and b_max) -> Strict interior search
    speculative_bs = np.arange(b_min + step, b_max, step)

    # Fallback safety if window is too narrow
    if len(speculative_bs) == 0:
        return float(round((b_min + b_max) / 2.0, 1))

    slopes_arr = np.array([c[0] for c in optimised_curves])[:, np.newaxis]
    intercepts_arr = np.array([c[1] for c in optimised_curves])[:, np.newaxis]

    # Shape: (N_curves, N_speculative_steps)
    log_k_grid = slopes_arr * speculative_bs + intercepts_arr
    n_curves = log_k_grid.shape[0]

    # Pairwise difference tensor shape: (N_curves, N_curves, N_speculative_steps)
    pairwise_diffs = np.abs(log_k_grid[:, np.newaxis, :] - log_k_grid[np.newaxis, :, :])

    # Mask out self-comparisons
    diag_mask = ~np.eye(n_curves, dtype=bool)

    # Extract off-diagonal peak pairs -> Shape: (N*(N-1), N_speculative_steps)
    off_diag_diffs = pairwise_diffs[diag_mask, :]

    # Minimum peak separation for each speculative %B step inside the interior window
    min_separations = off_diag_diffs.min(axis=0)

    # Select interior %B index that maximizes worst-case separation
    best_idx = int(np.argmax(min_separations))
    ideal_pct_b = float(speculative_bs[best_idx])

    print(f"[DEBUG int_pct_b] Interior Window: ({b_min}%, {b_max}%). Optimal Intermediate %B found: {ideal_pct_b}%")

    return float(round(ideal_pct_b, 1))


def handle_selectivity(df=None, prev_b=False):
    # 1. Initialize counter
    if 'selectivity_progression' not in st.session_state:
        st.session_state.selectivity_progression = 0
    
    st.session_state.selectivity_progression += 1
    
    # Initialize measured temperatures tracking array
    if 'all_measured_temps' not in st.session_state:
        st.session_state.all_measured_temps = []

    # Parse and store measured temperature from DataFrame if supplied
    if df is not None:
        measured_temp = extract_measured_temp(df)
        st.session_state.all_measured_temps.append(measured_temp)
    else:
        measured_temp = st.session_state.all_measured_temps[-1] if st.session_state.all_measured_temps else 25.0

    modified_parameters = []
    static_keys = [
        'all_solvents', 
        'all_ligands', 
        'all_bead_types',
        'all_pore_sizes',
        'all_coreshells', 
        'all_carbon_loads',
        'all_column_lengths',
        'all_internal_diameters', 
        'all_particle_sizes',
        'all_flow_rates', 
        'all_temperatures', 
        'all_pct_bs'
    ]
    
    # Under some circumstances switch to previous %B and continue changing selectivity
    if prev_b and len(st.session_state.all_pct_bs) >= 2:
        modified_parameters.append('all_pct_bs')
        st.session_state.all_pct_bs.append(st.session_state.all_pct_bs[-2])
        
    current_predicted_b = st.session_state.all_pct_bs[-1] if st.session_state.all_pct_bs else 50.0
    
    # Check current active solvent (e.g., 'meoh', 'me', 'methanol')
    current_solvent = str(st.session_state.all_solvents[-1]).strip().lower() if st.session_state.get('all_solvents') else 'acn'
    is_meoh = any(token in current_solvent for token in ['meoh', 'methanol', 'me'])

    match st.session_state.selectivity_progression:
        case 1:
            st.session_state.all_temperatures.append('40')
            modified_parameters.append('all_temperatures')
            
            # ONLY trigger temperature-adjusted Min %B UI if solvent is NOT MeOH / Me
            if not is_meoh:
                chosen_b = render_min_b_decision_ui(current_predicted_b, measured_temp, target_temp=40.0)
                if chosen_b != current_predicted_b:
                    st.session_state.all_pct_bs[-1] = chosen_b

            st.info(
                "**Next Run Strategy:** Increase Temperature to 40 °C to modify band spacing (selectivity) and reduce mobile phase viscosity. Shortens runtime and may reduce peak width. \n\n"
                "*Controlled by:* Column Oven | *Category:* Both (Chemical & Mechanical) | *Cycle:* Selectivity"
            )
            
        case 2:
            st.session_state.all_temperatures.append('60')
            modified_parameters.append('all_temperatures')
            
            # ONLY trigger temperature-adjusted Min %B UI if solvent is NOT MeOH / Me
            if not is_meoh:
                chosen_b = render_min_b_decision_ui(current_predicted_b, measured_temp, target_temp=60.0)
                if chosen_b != current_predicted_b:
                    st.session_state.all_pct_bs[-1] = chosen_b

            st.info(
                "**Next Run Strategy:** Increase Temperature to 60 °C to further adjust selectivity and sharpen peak shapes. Further shortens runtime and may reduce peak width. \n\n"
                "*Controlled by:* Column Oven | *Category:* Both (Chemical & Mechanical) | *Cycle:* Selectivity"
            )
            
        case 3:
            st.session_state.all_temperatures.append('amb')
            current_pct_b = st.session_state.all_pct_bs[-1]
            
            if is_meoh:
                new_solvent = 'ACN'
                nomogram_slope = 0.980221359
                nomogram_int = -6.246035141
                converted_b = current_pct_b * nomogram_slope + nomogram_int
            else:
                new_solvent = 'MeOH'
                nomogram_slope = 1.0090909
                nomogram_int = 6.9264069
                converted_b = current_pct_b * nomogram_slope + nomogram_int

            st.session_state.all_solvents.append(new_solvent)
            st.session_state.all_pct_bs.append(round(converted_b, 1))
            
            modified_parameters.extend(['all_temperatures', 'all_pct_bs', 'all_solvents'])
            
            st.info(
                f"**Next Run Strategy:** Swap Organic Solvent to {new_solvent} at nomogram-converted ratio ({round(converted_b, 1)}% B) at ambient temperature to alter solute-solvent solvophobic interactions like acidity, basicity and dipolarity at the same strength.\n\n"
                f"*Controlled by:* Pump / Solvents | *Category:* Chemical | *Cycle:* Selectivity"
            )
            
        case 4:
            st.session_state.all_temperatures.append('40')
            modified_parameters.append('all_temperatures')
            
            # ONLY trigger temperature-adjusted Min %B UI if solvent is NOT MeOH / Me
            if not is_meoh:
                chosen_b = render_min_b_decision_ui(current_predicted_b, measured_temp, target_temp=40.0)
                if chosen_b != current_predicted_b:
                    st.session_state.all_pct_bs[-1] = chosen_b

            st.info(
                "**Next Run Strategy:** Increase Temperature to 40 °C to modify band spacing (selectivity) and reduce mobile phase viscosity. Shortens runtime and may reduce peak width. \n\n"
                "*Controlled by:* Column Oven | *Category:* Both (Chemical & Mechanical) | *Cycle:* Selectivity"
            )
        case 5:
            st.session_state.all_temperatures.append('60')
            modified_parameters.append('all_temperatures')
            
            # ONLY trigger temperature-adjusted Min %B UI if solvent is NOT MeOH / Me
            if not is_meoh:
                chosen_b = render_min_b_decision_ui(current_predicted_b, measured_temp, target_temp=60.0)
                if chosen_b != current_predicted_b:
                    st.session_state.all_pct_bs[-1] = chosen_b

            st.info(
                "**Next Run Strategy:** Increase Temperature to 60 °C to further adjust selectivity and sharpen peak shapes. Further shortens runtime and may reduce peak width. \n\n"
                "*Controlled by:* Column Oven | *Category:* Both (Chemical & Mechanical) | *Cycle:* Selectivity"
            )
            
        case 6:
            default_keys = [
                'all_csvs', 'all_peak_counts', 'all_run_times', 'all_retention_times',
                'all_resolutions_overall', 'all_resolutions_final', 'all_equilibrated',
                'all_backpressures', 'all_pct_bs', 'all_solvents', 'all_ligands',
                'all_bead_types', 'all_pore_sizes', 'all_coreshells', 'all_carbon_loads',
                'all_column_lengths', 'all_internal_diameters', 'all_particle_sizes',
                'all_flow_rates', 'all_temperatures', 'all_sample_concentrations',
                'all_measured_temps'
            ]
            
            last_ligand = st.session_state.all_ligands[-1] if st.session_state.get('all_ligands') else "current"
            
            for key in default_keys:
                if key in st.session_state:
                    del st.session_state[key]
                    
            st.session_state.selectivity_progression = 0
            st.session_state.reset_message = (
                f"Ligand change required. Runs wiped because current history is non-transferable. "
                f"Please select a new stationary phase other than '{last_ligand}' using the top menu."
            )
            st.session_state.pending_ligand_change = True
            st.rerun()
            return

    # 2. Propagate unmodified parameters forward
    keep_the_same = [key for key in static_keys if key not in modified_parameters]
    for key in keep_the_same:
        if key in st.session_state and st.session_state[key]:
            st.session_state[key].append(st.session_state[key][-1])

def equipment_interactive():
    """
    Standalone interactive HPLC equipment & parameter monograph viewer.
    Features dual-column navigation for modules vs. parameters, segmented top overview,
    and side-by-side display for calculated vs. read chromatogram parameters.
    """

    # --------------------------------------------------------------------------
    # 2. DEFINITIONS & DICTIONARIES
    # --------------------------------------------------------------------------
    MODULE_KEYS = {"solvents", "degasser", "pump", "injector", "column", "column_oven", "detector"}
    PARAMETER_KEYS = {"selectivity", "retention", "efficiency", "backpressure", "area_under_peak", "peak_height", "peak_width", "runtime", "resolution"}

    LABEL_MAP = {
        "Instruments": "Instruments Overview",
        "solvents": "Solvents",
        "degasser": "Degasser",
        "pump": "Pump",
        "injector": "Injector",
        "column": "Column",
        "column_oven": "Column Oven",
        "detector": "Detector",
        "selectivity": "Selectivity",
        "retention": "Retention Time",
        "efficiency": "Efficiency (N)",
        "backpressure": "Backpressure",
        "area_under_peak": "Area Under Peak",
        "peak_height": "Peak Height",
        "peak_width": "Peak Width",
        "runtime": "Runtime",
        "resolution": "Resolution (Rs)",
        "parameters_read": "Parameters Read from Chromatogram",
        "parameters_calc": "Parameters Calculated"
    }

    REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

    NAVIGATION_LINKS = {
        "Instruments": [
            "solvents", "degasser", "pump", "injector", 
            "column", "column_oven", "detector", "parameters_read", "parameters_calc"
        ],
        "solvents": ["selectivity", "retention", "Instruments", "parameters_read", "parameters_calc"],
        "degasser": ["Instruments", "parameters_read", "parameters_calc"],
        "pump": ["retention", "efficiency", "backpressure", "Instruments", "parameters_read", "parameters_calc"],
        "injector": ["area_under_peak", "Instruments", "parameters_read", "parameters_calc"],
        "column": ["efficiency", "backpressure", "selectivity", "runtime", "Instruments", "parameters_read", "parameters_calc"],
        "column_oven": ["runtime", "backpressure", "selectivity", "retention", "Instruments", "parameters_read", "parameters_calc"],
        "detector": ["area_under_peak", "Instruments", "parameters_read", "parameters_calc"],
        # Parameters
        "selectivity": ["solvents", "pump", "column", "column_oven", "Instruments", "parameters_read", "parameters_calc"],
        "retention": ["solvents", "pump", "column_oven", "Instruments", "parameters_read", "parameters_calc"],
        "efficiency": ["pump", "column", "Instruments", "parameters_read", "parameters_calc"],
        "backpressure": ["pump", "column", "column_oven", "Instruments", "parameters_read", "parameters_calc"],
        "area_under_peak": ["injector", "detector", "Instruments", "parameters_read", "parameters_calc"],
        "peak_height": ["Instruments", "parameters_read", "parameters_calc"],
        "resolution": ["pump", "column", "column_oven", "solvents", "Instruments", "parameters_read", "parameters_calc"],
        "runtime": ["column", "column_oven", "pump", "Instruments", "parameters_read", "parameters_calc"],
        # System Overviews
        "parameters_read": [
            "peak_height", "area_under_peak", "efficiency", 
            "peak_width", "resolution", "backpressure", "runtime", "Instruments", "parameters_calc"
        ],
        "parameters_calc": [
            "resolution", "retention", "selectivity", "efficiency", "Instruments", "parameters_read"
        ]
    }

    if "active_equipment_node" not in st.session_state:
        st.session_state.active_equipment_node = "Instruments"

    current_node = st.session_state.active_equipment_node

    with st.container():
        valid_targets = NAVIGATION_LINKS.get(current_node, ["Instruments"])
        top_system_keys = ["Instruments", "parameters_read", "parameters_calc"]

        # Top System Overview Control
        top_keys = [k for k in top_system_keys if k in valid_targets or k == current_node]
        if current_node == "Instruments" and "Instruments" in top_keys:
            top_keys.remove("Instruments")
            
        top_labels = [LABEL_MAP.get(k, k) for k in top_keys]
        current_label = LABEL_MAP.get(current_node, current_node)

        if top_labels:
            selected_top = st.segmented_control(
                "System Overview Views:",
                options=top_labels,
                default=current_label if current_label in top_labels else None,
                selection_mode="single",
                key=f"seg_ctrl_top_{current_node}"
            )
            if selected_top and selected_top != current_label:
                st.session_state.active_equipment_node = REVERSE_LABEL_MAP.get(selected_top, "Instruments")
                st.rerun()

        st.markdown("---")

        # Two-Section Layout: Side Navigation (Left) + Monograph Display (Right)
        col_nav, col_display = st.columns([2, 3], gap="medium")

        with col_nav:
            # Separate available targets into Modules vs Parameters
            bottom_keys = [k for k in valid_targets if k not in top_system_keys]
            
            nav_modules = [k for k in bottom_keys if k in MODULE_KEYS]
            nav_parameters = [k for k in bottom_keys if k in PARAMETER_KEYS]

            # Split Navigation into two side-by-side sub-columns
            subcol_modules, subcol_params = st.columns(2, gap="small")

            with subcol_modules:
                st.caption("Hardware Modules")
                if nav_modules:
                    for key in nav_modules:
                        label = LABEL_MAP.get(key, key)
                        is_active = (key == current_node)
                        btn_type = "primary" if is_active else "secondary"
                        if st.button(label, key=f"v_nav_m_{key}", use_container_width=True, type=btn_type):
                            st.session_state.active_equipment_node = key
                            st.rerun()
                else:
                    st.write("—")

            with subcol_params:
                st.caption("Parameters")
                if nav_parameters:
                    for key in nav_parameters:
                        label = LABEL_MAP.get(key, key)
                        is_active = (key == current_node)
                        btn_type = "primary" if is_active else "secondary"
                        if st.button(label, key=f"v_nav_p_{key}", use_container_width=True, type=btn_type):
                            st.session_state.active_equipment_node = key
                            st.rerun()
                else:
                    st.write("—")

        with col_display:
            if current_node == "Instruments":
                st.subheader("HPLC System Instrumentation Overview")
                st.image("assets/manual_injector.png")
                st.image("assets/uhplc.png")
                st.image("assets/autosampler.png")
            elif current_node == "parameters_calc":
                st.subheader("Calculated Chromatographic Parameters")
                # 2x2 side-by-side display grid for calculated parameters
                grid_col1, grid_col2 = st.columns(2, gap="small")
                with grid_col1:
                    st.image("assets/resolution.png", caption="Resolution (Rs)")
                    st.image("assets/selectivity.png", caption="Selectivity (α)")
                with grid_col2:
                    st.image("assets/retention.png", caption="Retention Factor (k)")
                    st.image("assets/efficiency.png", caption="Efficiency / Column Plates (N)")
            else:
                node_title = LABEL_MAP.get(current_node, current_node)
                st.subheader(f"Monograph: {node_title}")
                st.image(f"assets/{current_node}.png")


def main():
    # --------------------------------------------------------------------------
    # Page Configuration & Header
    # --------------------------------------------------------------------------
    st.set_page_config(layout='wide')

    
    st.title("Ahmed's HPLC Method Development Engine")

    # ==============================================================================
    # USER INSTRUCTIONS (COLLAPSIBLE EXPANDER)
    # ==============================================================================
    with st.expander("📖 View Method Development Instructions & Workflow Guide", expanded=False):
        st.markdown(
            """
            Watch the video tutorial for a step-by-step walkthrough of how to use this app.
            https://youtu.be/BZxVgvxXyQs
            ### 1️⃣ Set System Specifications
            * Enter the specifications for your run, including backpressure, runtime, target number of peaks, and resolution requirements.
            * Click the "Submit specifications" button to save your settings before uploading any runs.
            * If you do not do this, then the default specifications will be submitted. They may not meet your run requirements.
            * Click on the Edit Column Inventory menu to change the inventory of available columns. Columns your lab has will be sorted to the top of the pile when efficiency is calculated.
            ---

            ### 2️⃣ Upload Chromatographic Runs Sequentially
            * Upload your run CSV files one by one as you execute them.
            * The engine automatically extracts run conditions and physical column specifications directly from your filenames.
            * Your first run will be required to establish the initial conditions for the process. 
            * It will automatically be read from the filename if it follows the naming convention, or you can manually input the conditions.
            * If you would like to manually input the conditions, go to the top of the page and input the conditions in the "Method Specifications" section.
            * Then, submit the conditions with the 'Submit first run conditions' button before uploading your first run.
            * Do NOT submit first run conditions after uploading your first run, as this will overwrite the extracted conditions from the filename.
            * Upload one run at a time, and the engine will advise the conditions for the next run based on the results of the previous runs.
            * Use the (?) icon next to each input field for additional information about the parameter.

            ---

            ### 3️⃣ Automated Conditions
            * Minimum %B is calculated using linear regression of $\log_{10}(k)$ vs. %B. of the last peak in your run.
            * You can choose to adjust the minimimum %B based on the temperature of the next run, which will be used to modify the selectivity of your method.
            * Minimum %B is calculated only for the first solvent. An isoeluotropic concentration will be recommended if the solvent changes.
            * Intermediate %B is calculated by linear regression of $\log_{10}(k)$ vs. %B for all peaks in your run.
            * Initially, Intermediate %B builds a rough model of each compound using elution order. Using Hungarian algorithm, peaks are assigned accordingly.
            * Due to divergence and high-%B runs having few peaks, the model restricts the calculation to the likely %B ranges that will have an ideal, missed %B.
            * Scroll down to view the Intermediate %B and Selectivity Plot.
            * Efficiency calculations are automated entirely. Preference is given to runs that change one parameter and still meet specification, due to simplicity.
            * The tool returns all admissible conditions for the next run if the prescribed strategy is not sufficient.
            ---

            ### 4️⃣ Review Plot & Strategy Selection
            * **Selectivity Plot:** Inspect $\log_{10}(k)$ vs. %B linear regression curves (legend displays $R^2$ to 6 decimal places).
            * **Radio Decision Options:** Where multiple valid run paths exist, select your target condition using the provided radio buttons or a segmented control.
            ---
            
            ### 5️⃣ Equipment Interactive Viewer
            * Explore the HPLC system using the interactive equipment viewer.
            * The left column provides navigation for hardware modules and chromatographic parameters, while the right column displays detailed monographs with images and descriptions.
            * Every run tells you which module is controlling the parameter, and how it is controlled (chemical, mechanical, or both).
            """
        )
        st.info(
            "💡 **Best Practice:** Execute initial screening runs at clean 10% organic solvent increments "
            "(e.g., 30%, 40%, 50%) for optimal linear fit accuracy."
        )
        st.text('Edit the specifications and run conditions to match your method. An example is provided. Hover over the (?) icon for information. Make sure to submit your changes.')
        st.text('Additionally, you may automatically input the CSV of your first run to fill out the run conditions automatically.')
    print('''chromatogram_file = st.file_uploader("Upload Chromatogram CSV", type=["csv"])

    if chromatogram_file is not None:
        # Check if first run conditions need initialization
        if 'all_pct_bs' not in st.session_state or len(st.session_state.all_pct_bs) == 0:
            
            mode = st.radio(
                "Select Initial Conditions Input Mode:",
                ["Automatic (Parse from Filename)", "Manual Input"],
                horizontal=True
            )

            if mode == "Automatic (Parse from Filename)":
                # Extract base filename without extension
                filename_str = chromatogram_file.name.rsplit('.', 1)[0]
                parts = filename_str.split('-')

                if len(parts) >= 14:
                    # Direct string parsing according to naming convention
                    solvent_auto = parts[3]
                    percent_b_auto = float(parts[6])
                    flow_rate_auto = float(parts[7])
                    ligand_auto = parts[11]
                    temp_auto = parts[13]

                    # Parse dimensions string (e.g. 150x4.6x5)
                    dim_parts = parts[12].lower().split('x')
                    length_auto = float(dim_parts[0])
                    internal_auto = float(dim_parts[1])
                    particle_size_auto = float(dim_parts[2])

                    # Default physical column properties
                    particle_type_auto = "Fully Porous"
                    poresize_auto = "100Å"
                    coreshell_auto = "No"
                    carbon_load_auto = "High"

                    st.success("Successfully parsed initial conditions from filename:")
                    st.write({
                        "%B": percent_b_auto, "Solvent": solvent_auto, "Ligand": ligand_auto,
                        "Temp": temp_auto, "Flow Rate": flow_rate_auto, 
                        "Dimensions (L x ID x dp)": f"{length_auto} x {internal_auto} x {particle_size_auto}"
                    })

                    if st.button("Confirm automatic conditions for the first run", type="primary"):
                        st.session_state.all_pct_bs.append(percent_b_auto)
                        st.session_state.all_solvents.append(solvent_auto)
                        st.session_state.all_ligands.append(ligand_auto)
                        st.session_state.all_bead_types.append(particle_type_auto)
                        st.session_state.all_pore_sizes.append(poresize_auto)
                        st.session_state.all_coreshells.append(coreshell_auto)
                        st.session_state.all_carbon_loads.append(carbon_load_auto)
                        st.session_state.all_column_lengths.append(length_auto)
                        st.session_state.all_internal_diameters.append(internal_auto)
                        st.session_state.all_particle_sizes.append(particle_size_auto)
                        st.session_state.all_flow_rates.append(flow_rate_auto)
                        st.session_state.all_temperatures.append(temp_auto)
                        st.rerun()

                else:
                    st.error("Uploaded filename does not match the expected naming format. Please switch to Manual Input.")
    ''')
    # --------------------------------------------------------------------------
    # Main Layout Split (2 Columns)
    # --------------------------------------------------------------------------
    leftside, rightside = st.columns(2)
    
    # ligand change notification
    if st.session_state.pending_ligand_change == True:
        st.warning('The ligand needs to change after the last run. Please select a new ligand and make sure to submit your changes.')
        if st.button('I understand'): 
            st.session_state.pending_ligand_change == False
            st.rerun()

    # ==========================================================================
    # LEFT COLUMN: Specifications & Run Conditions
    # ==========================================================================
    with leftside:
        with st.container(border=True):
            # --- Section: Method Specifications ---
            st.header('Method Specifications')
            
            peakspec = st.number_input(
                'Target Number of Peaks', 
                min_value=1, 
                value=7, 
                help='At least one peak'
            )
            
            runspec = st.number_input(
                'Run Time Specification (min)', 
                min_value=0.0, 
                value=15.0, 
                help='At least 0 minutes'
            )
            
            res_spec = st.number_input(
                'Resolution', 
                min_value=0.0, 
                value=1.4
            )
            
            bpspec = st.number_input(
                'Back Pressure Specification Limit (psi)', 
                value=4000.0, 
                help='Safety backpressure limit on HPLC'
            )
            st.caption("Controlled by column resistance to the pump's controlled flow rate.")

            # --- Section: First Run Conditions ---
            st.header('First Run Conditions')
            
            # Chemical Parameters
            st.subheader('Chemical Parameters')
            
            solventf = st.selectbox('Solvent', ['ACN', 'MeOH'])
            st.caption('Solvent is controlled by choice on the machine and the attached solvent line.')
            
            percent_bf = st.number_input(
                '% Organic Solvent Concentration (%B)', 
                min_value=0.0, 
                max_value=100.0, 
                value=100.0, 
                help='Between 0% and 100%'
            )
            st.caption("%B is controlled by the pump's proportionation of solvents.")

            # Mechanical Parameters
            st.subheader('Mechanical Parameters')
            
            sample_cf = st.number_input('Sample Concentration (mg/mL)', min_value=0.0, value=0.1)
            
            flow_ratef = st.number_input('Flow Rate (mL/min)', min_value=0.0, value=1.5)
            st.caption('Flow rate is controlled by the pump.')
            
            wavelength = st.number_input('Wavelength (nm)', min_value=190.0, max_value=950.0, value=254.0)
            st.caption('Wavelength is controlled by the detector.')

            # Combined Parameters (Temperature Selection)
            st.subheader('Both Mechanical & Chemical Parameter')
            
            is_ambient = st.radio(
                'Choose whether the first run will be at ambient temperature or not.', 
                ['Ambient', 'Not Ambient']
            )
            
            if is_ambient.lower().strip() != 'ambient':
                tempf = st.number_input(
                    'Temperature (ºC)', 
                    min_value=0.0, 
                    max_value=80.0, 
                    help='Between 0º C and 80º C for column safety.'
                )
                if tempf >= 60:
                    st.info('Ensure that the temperature is below the limit of the column.')
            else:
                tempf = 'ambient'
                st.text('Ambient Temperature Selected')
                
            st.caption('Temperature is controlled by the column oven.')

    # ==========================================================================
    # RIGHT COLUMN: Column Properties & Packing
    # ==========================================================================
    with rightside:
        with st.container(border=True):
            # --- Section: Column Dimensions ---
            st.header('Column Dimensions')
            st.caption('These parameters are controlled by the column selection.')
            
            st.subheader('Mechanical Parameters')
            lengthf = st.number_input('Column Length (mm)', min_value=0.0, value=150.0)
            internalf = st.number_input('Internal Diameter (mm)', min_value=0.0, value=4.6)
            particle_sizef = st.number_input('Particle Size (µM)', min_value=0.0, value=5.0)

            # --- Section: Column Packing ---
            st.header('Column Packing')
            
            st.subheader('Chemical parameters')
            ligandf = st.text_input('Ligand', value='C18P')
            carbon_loadf = st.number_input(
                'Carbon Load (%)', 
                min_value=0.0, 
                max_value=100.0, 
                value=11.0, 
                help='Between 0% and 100%.'
            )

            st.subheader('Mechanical Parameters')
            particle_typef = st.text_input('Bead Type', value='FPP')
            coreshellf = st.checkbox('Coreshell?', value=False)
            poresizef = st.number_input('Pore Size (Å)', min_value=0.0, value=140.0)

    # Uploading & Processing of a file
    default_keys = [
        'all_csvs',
        'all_peak_counts',
        'all_run_times',
        'all_retention_times',
        'all_resolutions_overall',
        'all_resolutions_final',
        'all_equilibrated',
        'all_backpressures',
        'all_pct_bs',
        'all_solvents',
        'all_ligands',
        'all_bead_types',
        'all_pore_sizes',
        'all_coreshells',
        'all_carbon_loads',
        'all_column_lengths',
        'all_internal_diameters',
        'all_particle_sizes',
        'all_flow_rates',
        'all_temperatures',
        'all_sample_concentrations',
        'minbdone'
    ]

    for key in default_keys:
        if key not in st.session_state:
            st.session_state[key] = []
    
    # Adding the first run values
    if st.button('Submit first run conditions (manual mode, if your file naming fails to be parsed)'):
        st.session_state.all_pct_bs.append(percent_bf)
        st.session_state.all_solvents.append(solventf)
        st.session_state.all_ligands.append(ligandf)
        st.session_state.all_bead_types.append(particle_typef)
        st.session_state.all_pore_sizes.append(poresizef)
        st.session_state.all_coreshells.append(coreshellf)
        st.session_state.all_carbon_loads.append(carbon_loadf)
        st.session_state.all_column_lengths.append(lengthf)
        st.session_state.all_internal_diameters.append(internalf)
        st.session_state.all_particle_sizes.append(particle_sizef)
        st.session_state.all_flow_rates.append(flow_ratef)
        st.session_state.all_temperatures.append(tempf)

    if 'submitted_specs' not in st.session_state:
        st.session_state['submitted_specs'] = False

    if st.button('Submit specifications.'):
        st.session_state.submitted_specs = True
        st.session_state.peakspec = peakspec
        st.session_state.runspec = runspec
        st.session_state.res_spec = res_spec
        st.session_state.bpspec = bpspec
    # Tracker set to prevent re-parsing on every Streamlit page re-render
    if 'processed_file_ids' not in st.session_state:
        st.session_state.processed_file_ids = set()

    # Run Conditions Reminder
    render_next_run_conditions_preview()
  
    # 2. Uploading & Single-Execution Processing
    added_file = st.file_uploader('Upload a CSV file of your run.', type='csv', key="main_run_uploader")

    if added_file is not None:
        # Unique file fingerprint combining filename and byte size
        file_id = f"{added_file.name}_{added_file.size}"
        
        # Process ONLY if this specific upload hasn't been parsed yet
        if file_id not in st.session_state.processed_file_ids:
            # Barrier for misuse (Specs not submitted)
            if not st.session_state.submitted_specs:
                st.session_state.peakspec = peakspec
                st.session_state.runspec = runspec
                st.session_state.res_spec = res_spec
                st.session_state.bpspec = bpspec
            try:
                dataframe_added = safe_read(added_file)

                # Peak Count
                peak_cnt = peak_count(dataframe_added)
                
                # Run Time & Retention Times
                retention_analysis = retention_time(dataframe_added, peak_cnt)
                
                # Resolution
                resolution_analysis = find_resolution(dataframe_added, peak_cnt)
                
                # Backpressure
                bp_analysis = find_backpressure(dataframe_added)
                
                
                # --- FILENAME PARSING ---
                filename_str = added_file.name.rsplit('.', 1)[0]
                parts = filename_str.split('-')
                
                # Determine current run index
                run_idx = len(st.session_state.all_csvs)
                
                if len(parts) >= 14:
                    actual_solvent = parts[3].strip()
                    actual_pct_b = float(parts[6])
                    actual_flow = float(parts[7])
                    actual_ligand = parts[11].strip()
                    actual_temp = str(parts[13]).strip()
                    
                    # Parse dimensions (e.g., 150x4.6x5)
                    dim_parts = parts[12].lower().split('x')
                    actual_length = float(dim_parts[0]) if len(dim_parts) > 0 else 150.0
                    actual_id = float(dim_parts[1]) if len(dim_parts) > 1 else 4.6
                    actual_particle_size = float(dim_parts[2]) if len(dim_parts) > 2 else 5.0

                    # Standard defaults for physical column properties
                    actual_bead_type = "FPP"
                    actual_pore_size = 140.0
                    actual_coreshell = False
                    actual_carbon_load = 11.0
                    
                    # ----------------------------------------------------------
                    # CASE A: FIRST RUN INITIALIZATION (run_idx == 0)
                    # ----------------------------------------------------------
                    if run_idx == 0 or len(st.session_state.all_pct_bs) == 0:
                        st.session_state.all_pct_bs.append(actual_pct_b)
                        st.session_state.all_solvents.append(actual_solvent)
                        st.session_state.all_ligands.append(actual_ligand)
                        st.session_state.all_temperatures.append(actual_temp)
                        st.session_state.all_flow_rates.append(actual_flow)
                        st.session_state.all_column_lengths.append(actual_length)
                        st.session_state.all_internal_diameters.append(actual_id)
                        st.session_state.all_particle_sizes.append(actual_particle_size)
                        st.session_state.all_bead_types.append(actual_bead_type)
                        st.session_state.all_pore_sizes.append(actual_pore_size)
                        st.session_state.all_coreshells.append(actual_coreshell)
                        st.session_state.all_carbon_loads.append(actual_carbon_load)
                        
                        st.info(
                            f"🚀 **First Run Initialized from Filename:** %B: {actual_pct_b}%, "
                            f"Solvent: {actual_solvent}, Column: {actual_ligand} ({actual_length}x{actual_id}mm, {actual_particle_size}µm), "
                            f"Temp: {actual_temp}°C, Flow: {actual_flow} mL/min"
                        )
                    
                    # ----------------------------------------------------------
                    # CASE B: SUBSEQUENT RUNS & DEVIATION CHECK (run_idx > 0)
                    # ----------------------------------------------------------
                    else:
                        # Fetch expected conditions for this specific run index
                        expected_pct_b = st.session_state.all_pct_bs[run_idx] if run_idx < len(st.session_state.all_pct_bs) else actual_pct_b
                        expected_solvent = str(st.session_state.all_solvents[run_idx]).strip() if run_idx < len(st.session_state.all_solvents) else actual_solvent
                        expected_ligand = str(st.session_state.all_ligands[run_idx]).strip() if run_idx < len(st.session_state.all_ligands) else actual_ligand
                        expected_temp = str(st.session_state.all_temperatures[run_idx]).strip() if run_idx < len(st.session_state.all_temperatures) else actual_temp
                        actual_temp_num = normalize_temp_value(actual_temp)
                        expected_temp_num = normalize_temp_value(expected_temp)

                        has_temp_dev = abs(actual_temp_num - expected_temp_num) > 0.5
                        
                        # Check for deviations against expectations
                        has_pct_b_dev = (actual_pct_b != expected_pct_b)
                        has_solvent_dev = (actual_solvent.upper() != expected_solvent.upper())
                        has_ligand_dev = (actual_ligand.upper() != expected_ligand.upper())
                        has_temp_dev = (actual_temp.upper() != expected_temp.upper())
                        
                        if has_pct_b_dev or has_solvent_dev or has_ligand_dev or has_temp_dev:
                            st.warning(
                                f"User deviation from method development framework for Run {run_idx + 1}. Actual conditions are:\n"
                                f"- %B: {actual_pct_b}% (Expected: {expected_pct_b}%)\n"
                                f"- Solvent: {actual_solvent} (Expected: {expected_solvent})\n"
                                f"- Ligand: {actual_ligand} (Expected: {expected_ligand})\n"
                                f"- Temperature: {actual_temp}°C (Expected: {expected_temp}°C)"
                            )
                            
                            # Sync selectivity progression case if solvent/temp/ligand changed
                            if has_solvent_dev or has_temp_dev or has_ligand_dev:
                                if 'selectivity_progression' not in st.session_state:
                                    st.session_state.selectivity_progression = 0
                                initial_solvent = st.session_state.all_solvents[0] if st.session_state.all_solvents else actual_solvent
                                matched_case = determine_selectivity_case(actual_solvent, actual_temp, initial_solvent)
                                st.session_state.selectivity_progression = matched_case
                        
                        # Store/Correct ACTUAL conditions at index [run_idx]
                        if run_idx < len(st.session_state.all_pct_bs):
                            st.session_state.all_pct_bs[run_idx] = actual_pct_b
                        else:
                            st.session_state.all_pct_bs.append(actual_pct_b)
                            
                        if run_idx < len(st.session_state.all_solvents):
                            st.session_state.all_solvents[run_idx] = actual_solvent
                        else:
                            st.session_state.all_solvents.append(actual_solvent)

                        if run_idx < len(st.session_state.all_ligands):
                            st.session_state.all_ligands[run_idx] = actual_ligand
                        else:
                            st.session_state.all_ligands.append(actual_ligand)
                            
                        if run_idx < len(st.session_state.all_temperatures):
                            st.session_state.all_temperatures[run_idx] = actual_temp
                        else:
                            st.session_state.all_temperatures.append(actual_temp)

                # Append parsed chromatogram values to session state
                st.session_state.all_csvs.append(dataframe_added)
                st.session_state.all_peak_counts.append(peak_cnt)
                st.session_state.all_retention_times.append(retention_analysis['retention_times'])
                st.session_state.all_run_times.append(retention_analysis['runtime'])
                st.session_state.all_resolutions_overall.append(resolution_analysis['resolutions'])
                st.session_state.all_resolutions_final.append(resolution_analysis['min_res'])
                st.session_state.all_equilibrated.append(bp_analysis['equilibrated'])
                st.session_state.all_backpressures.append(bp_analysis['backpressure'])

                efficiency_data = Efficiency.find_efficiency(dataframe_added)
                required_n = Efficiency.predict_required_plates()

                # Mark file as processed
                st.session_state.processed_file_ids.add(file_id)
                st.success(f"Successfully processed {added_file.name}!")
                
                # Evaluate next strategy automatically
                if st.session_state.all_csvs:
                    evaluate_next_run_conditions(peakspec, runspec, res_spec, bpspec)

            except Exception as e:
                tb_str = traceback.format_exc()
                print("=== FULL EXCEPTION TRACEBACK ===")
                print(tb_str)
                st.error(f"Error parsing {added_file.name}: {e}")
    # Optional View
    with st.expander("🔍 View Raw Parsed Values (Session State)", expanded=False):
        if len(st.session_state.all_csvs) == 0:
            st.info("No files have been parsed yet. Upload a CSV to view extracted data.")
        else:
            print(dict(st.session_state).keys())
            # Display the stored lists directly as raw session state values
            st.write("**Peak Counts:**", st.session_state.all_peak_counts)
            st.write("**Run Times:**", st.session_state.all_run_times)
            st.write("**Retention Times (per run):**", st.session_state.all_retention_times)
            st.write("**Resolutions Overall:**", st.session_state.all_resolutions_overall)
            st.write("**Resolutions Final (Min Res):**", st.session_state.all_resolutions_final)
            st.write("**Equilibrated:**", st.session_state.all_equilibrated)
            st.write("**Backpressures:**", st.session_state.all_backpressures)
            
            # Optional raw dictionary dump
            st.divider()
            st.caption("Raw Session State Object:")
            st.json({
                "peak_counts": st.session_state.all_peak_counts,
                "run_times": st.session_state.all_run_times,
                "retention_times": st.session_state.all_retention_times,
                "resolutions_overall": st.session_state.all_resolutions_overall,
                "resolutions_final": st.session_state.all_resolutions_final,
                "equilibrated": st.session_state.all_equilibrated,
                "backpressures": st.session_state.all_backpressures,
                "percent bs": st.session_state.all_pct_bs
            })
            if st.button('View raw session state'):
                st.json(st.session_state)
    
                    

    # In your main script where handle_efficiency is triggered:

    # Initialize persistent UI visibility and candidate state flags
    if "show_efficiency_ui" not in st.session_state:
        st.session_state.show_efficiency_ui = False
    if "efficiency_candidates" not in st.session_state:
        st.session_state.efficiency_candidates = []

    # Persistent UI Block: Executes directly in main()
    if st.session_state.show_efficiency_ui and st.session_state.efficiency_candidates:
        candidates = st.session_state.efficiency_candidates

        # Single Candidate Auto-Commit
        if len(candidates) == 1:
            Efficiency.commit_efficiency_selection(candidates[0])
            st.rerun()

        # Multi-Candidate Selection UI
        st.markdown("### ⚙️ **Select Next Run Conditions**")
        st.write(f"Identified **{len(candidates)} valid optimization options**:")

        options_map = {}
        for i, c in enumerate(candidates):
            label = (
                f"Option {i+1}: {', '.join(c['changes'])}\n"
                f"⏱️ tR: {c['tR']:.2f} min | "
                f"🎯 Rs: {c['predicted_rs']:.2f} | "
                f"⚡ BP: {c['bp']:.0f} psi | "
                f"🧪 Solvent: {c['solvent_used']:.1f} mL"
            )
            options_map[label] = c

        radio_key = "eff_radio_main_selection"

        selected_label = st.radio(
            label="Choose optimization strategy:",
            options=list(options_map.keys()),
            index=0,
            key=radio_key
        )

        active_cand = options_map[selected_label]
        st.caption(f"💡 **Economic Justification:** {active_cand['justification']}")

        if st.button("Confirm Selected Strategy", type="primary"):
            Efficiency.commit_efficiency_selection(active_cand)
            st.rerun()

    # Persistent Decision Radio Block (outside st.button to maintain state)
    if st.session_state.get('show_decision_radio', False):
        choice1 = st.session_state.get('choice1', 50.0)
        choice2 = st.session_state.get('choice2', 50.0)
        
        # Safely compute expected runtime for Intermediate %B
        slope = st.session_state.get('tr_slope', 0.0)
        intercept = st.session_state.get('tr_intercept', 0.0)
        tzero = st.session_state.get('tr_tzero', 0.0)
        
        try:
            expected_intb_logk = choice2 * slope + intercept
            expected_intb_tr = (10 ** expected_intb_logk) * tzero + tzero
            runtime_display = f"{ceil(expected_intb_tr)} min" if expected_intb_tr > 0 else "N/A"
        except Exception:
            runtime_display = "N/A"

        choice_made = st.radio(
            f"Select whether to proceed with Intermediate %B ({choice2}%), or Minimum %B ({choice1}%):\n\n"
            f"Expected runtime at Intermediate %B: {runtime_display} | Expected runtime at Minimum %B: {runspec} min",
            ['Not Selected', 'Min%B', 'Intermediate %B'],
            key='decision_radio_selection'
        )

        if st.button('Confirm %B Selection'):
            if choice_made == 'Not Selected':
                st.warning("Please select either 'Min%B' or 'Intermediate %B' before confirming.")
            else:
                selected_b = choice1 if choice_made == 'Min%B' else choice2
                
                # --- DEFINITIVE ARRAY SYNCHRONIZATION ---
                # Define all 12 tracked static parameters
                all_keys = [
                    'all_pct_bs', 'all_solvents', 'all_ligands', 'all_bead_types',
                    'all_pore_sizes', 'all_coreshells', 'all_carbon_loads',
                    'all_column_lengths', 'all_internal_diameters', 'all_particle_sizes',
                    'all_flow_rates', 'all_temperatures'
                ]
                
                # 1. Modify explicitly changed key (%B)
                st.session_state.all_pct_bs.append(selected_b)
                modified_parameters = ['all_pct_bs']
                
                # 2. Replicate all unmodified keys strictly ONCE
                for key in all_keys:
                    if key not in modified_parameters:
                        if key in st.session_state and st.session_state[key]:
                            st.session_state[key].append(st.session_state[key][-1])

                st.info(
                    f"**Next Run Strategy:** Change %B to {selected_b}%\n\n"
                    f"*Controlled by:* Pump | *Parameter Type*: Chemical | *Cycle:* Retention"
                )
                st.success(f"Added %B condition: {selected_b}% for next run!")
                
                # Reset UI trigger
                st.session_state.show_decision_radio = False
                st.rerun()
        
    if st.session_state.get('all_backpressures') and len(st.session_state.all_backpressures) > 0:
        st.subheader("📜 Historical Run Record & Decisions")

        # Build history records list safely
        history_records = []
        total_runs = len(st.session_state.all_backpressures)

        for i in range(total_runs):
            run_no = i + 1
            
            # Safely pull conditions per run index
            pct_b = st.session_state.all_pct_bs[i] if i < len(st.session_state.all_pct_bs) else "-"
            solvent = st.session_state.all_solvents[i] if i < len(st.session_state.all_solvents) else "-"
            temp = st.session_state.all_temperatures[i] if i < len(st.session_state.all_temperatures) else "-"
            ligand = st.session_state.all_ligands[i] if i < len(st.session_state.all_ligands) else "-"
            flow = st.session_state.all_flow_rates[i] if i < len(st.session_state.all_flow_rates) else "-"

            # Measured responses
            bp = st.session_state.all_backpressures[i] if i < len(st.session_state.all_backpressures) else 0.0
            peaks = st.session_state.all_peak_counts[i] if i < len(st.session_state.all_peak_counts) else 0
            runtime = st.session_state.all_run_times[i] if i < len(st.session_state.all_run_times) else 0.0
            res = st.session_state.all_resolutions_final[i] if i < len(st.session_state.all_resolutions_final) else 0.0

            # Deriving the historical decision associated with the run state
            if bp > bpspec:
                decision = "⚠️ Overpressure Limit Exceeded"
            elif peaks > peakspec:
                decision = "⚠️ Extra Peaks (> Target) — Check Contamination"
            elif peaks < peakspec:
                if runtime < (0.5 * runspec):
                    decision = f"Retention Cycle: Decreased %B to {pct_b}% (-10%)"
                elif runtime < runspec:
                    decision = "Retention Optimization: Evaluated Nomogram/Intermediate %B"
                else:
                    decision = "Retention Timeout: Returned to Prev %B & Triggered Selectivity"
            elif peaks == peakspec:
                if runtime <= runspec and res >= res_spec:
                    decision = "🎉 Method Complete (All Specifications Met)"
                else:
                    decision = "Selectivity Frozen → Handed Off to Efficiency (Flow Chart 2)"
            else:
                decision = "Evaluated"

            history_records.append({
                "Run #": f"Run {run_no}",
                "%B Ratio": f"{pct_b}%",
                "Solvent": solvent,
                "Temp (°C)": temp,
                "Ligand": ligand,
                "Flow Rate": f"{flow} mL/min",
                "Backpressure": f"{bp:.1f} bar",
                "Peaks": f"{peaks} / {peakspec}",
                "Run Time": f"{runtime:.2f} min",
                "Resolution": f"{res:.2f}",
                "Associated Decision / Action": decision
            })

        # Render non-interactive summary dataframe
        df_history = pd.DataFrame(history_records)

        st.dataframe(
            df_history,
            width='stretch',
            hide_index=True
        )
    else:
        st.caption("No historical run records available yet. Upload a CSV chromatogram run to populate the record.")

   
    google_sheet_url = "https://docs.google.com/spreadsheets/d/1-Kl4UFp9Kk7IYOftbTtGVi-bior72YgTK0jYJZEbVK8/edit?gid=268052302#gid=268052302"

    
    st.header('log(K) vs %B for all peaks')
    st.text('Selectivity Plot (log10 k vs %B) is generated below if intermediate %B has been calculated.')
    if st.session_state.get('all_csvs'):
        fig = generate_selectivity_plot()
        if fig is not None:
            with st.expander("📈 Selectivity Plot (log10 k vs %B)", expanded=True):
                st.plotly_chart(fig, use_container_width='stretch', key=f"selectivity_plot_{len(st.session_state.all_csvs)}")

    if st.button('Recalculate Intermediate %B and Selectivity Plot'):
        # Execute optimized linear fit & Hungarian assignment
        new_int_b = safe_int_pct_b()
        st.success(f"Recalculated Intermediate %B: {new_int_b}%")
        print(f'I am reaching here: {new_int_b}')
    st.header('Equipment Interactive: HPLC')
    #st.markdown('<div id="equipment-viewer"></div>', unsafe_allow_html=True)
    equipment_interactive()
    st.caption(f'Want to view original source for equipment interactive?: {google_sheet_url}')

# ==============================================================================
# ENTRY POINT (Allows execution via double-click or CLI)
# ==============================================================================
if __name__ == "__main__":
    if not st.runtime.exists():
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())
    main()
