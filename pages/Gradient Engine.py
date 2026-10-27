import sys
from pathlib import Path
import os

# Add the pages directory to Python's module lookup path
sys.path.append(os.path.dirname(__file__))

from Engine import safe_read, peak_count, find_resolution, retention_time
import streamlit as st
from types import SimpleNamespace as sns
import pandas as pd
import numpy as np
from scipy.stats import linregress
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Custom SVG encoded as a data URI
b_profile_svg = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='12' fill='%23FFFFFF'/%3E%3Cpath d='M 8 48 L 20 48 L 40 16 L 50 16 L 56 48' fill='none' stroke='%23001F54' stroke-width='6' stroke-linecap='round' stroke-linejoin='round'/%3E%3Ccircle cx='20' cy='48' r='3.5' fill='%23D32F2F'/%3E%3Ccircle cx='40' cy='16' r='3.5' fill='%23D32F2F'/%3E%3C/svg%3E"

st.set_page_config(
    page_title="Gradient Engine",
    page_icon=b_profile_svg,
    layout="wide"
)

print('''
Case	Peaks	Rs min	Action
G1 ↑	< 7	0	Increase GS: same %B window, tG × 0.5
G1 ↓	< 7 (after G1 ↑)	0	Decrease GS: back to the scouting run, tG × 1.2
G2	= 7	< 1.4	Keep GS: F × 0.8 and tG ÷ 0.8
G3	= 7	≥ 1.4 but slow / wasted %B	Keep GS, shrink: trim the unused %B and cut tG in proportion
G4	= 7	≥ 1.4 and tR ≤ run-time spec	Method met: lock it 🎉
G1 ✕	< 7 (after G1 ↑ and ↓)	0	Change selectivity: temperature, solvent or ligand
''')
# Equation: GS = ΔΦ × Vm ÷ (tG × F)

def initialise_variables() -> None:
    # 1. Keys that default to None
    static_keys = [
        'grad_runspec',
        'grad_bpspec',
        'grad_res_spec',  # Added missing comma
        'grad_peakspec',
        'grad_F',
        'grad_c1',
        'grad_c2',
        'grad_tG',
        'grad_solvent',
        'grad_ligand',
        'grad_temperature',
        'delta_tG'
    ]
    for key in static_keys:
        if key not in st.session_state:
            st.session_state[key] = None

    # 2. List containers (MUST be initialized to [] rather than None)
    list_keys = [
        'grad_files',
        'grad_expected_run',
        'grad_conditions'
    ]
    for key in list_keys:
        if key not in st.session_state or st.session_state[key] is None:
            st.session_state[key] = []

    # 3. Numeric counters
    if 'g1_progression' not in st.session_state:
        st.session_state['g1_progression'] = 1

    if 'selectivity_progression' not in st.session_state:
        st.session_state['selectivity_progression'] = 0

    return None


def display_gradient_run_structure_expander(
    df: pd.DataFrame, current_sns: sns, next_sns: sns = None
) -> None:
    """Renders clean, styled HPLC chromatogram and method %B visualizations with

    strict axis formatting, scaled backpressure intensity, and explicit pressure axis bounds.
    """
    with st.expander("📈 **Run Structure & Chromatogram Visualizer**", expanded=True):
        tab_current, tab_next = st.tabs(["Current Run Analysis", "Next Optimized Run"])

        # Session state fallbacks
        if current_sns is None and "grad_conditions" in st.session_state:
            current_sns = st.session_state.grad_conditions[-1]

        if next_sns is None and "grad_expected_run" in st.session_state:
            next_sns = st.session_state.grad_expected_run[-1]

        # Calculate Void/Dwell Delay (Vm / F)
        radius_cm = (getattr(current_sns, 'id', 4.6) / 10.0) / 2.0
        length_cm = getattr(current_sns, 'L', 150.0) / 10.0
        v_m = np.pi * (radius_cm**2) * length_cm * 0.68
        F = getattr(current_sns, 'F', 1.0)
        t_delay = v_m / F if F > 0 else 0.0

        FONT_FAMILY = "Arial, Calibri, sans-serif"
        COLOR_NAVY = "#001F54"
        COLOR_BLACK = "#000000"
        COLOR_PRESSURE = "#D32F2F"  # Red pressure trace line
        COLOR_GRID = "#E2E8F0"      # Light Subtle Grey Grid

        # =====================================================================
        # TAB 1: CURRENT RUN (Chromatogram + Backpressure + %B Profile)
        # =====================================================================
        with tab_current:
            first_col = df.iloc[:, 0].astype(str)

            # 1. Parse Chromatogram UV Block
            has_chrom = first_col.str.contains('chromatogram', case=False, na=False)
            has_det = first_col.str.contains('detector', case=False, na=False)
            has_no_press = ~first_col.str.contains('pressure', case=False, na=False)
            has_no_temp = ~first_col.str.contains('temp', case=False, na=False)

            chrom_matches = has_chrom & has_det & has_no_press & has_no_temp
            if chrom_matches.any():
                chrom_start = chrom_matches.values.argmax() + 8
                has_press_after = first_col.str.contains(
                    'pressure', case=False, na=False
                ) & (first_col.index > chrom_start)
                chrom_end = (
                    has_press_after.values.argmax()
                    if has_press_after.any()
                    else len(df)
                )

                uv_df = df.iloc[chrom_start:chrom_end, :2].dropna().copy()
                uv_df.columns = ['time', 'uv']
                uv_df['time'] = pd.to_numeric(uv_df['time'], errors='coerce')
                uv_df['uv'] = pd.to_numeric(uv_df['uv'], errors='coerce')
                uv_df = uv_df.dropna().reset_index(drop=True)
            else:
                uv_df = pd.DataFrame({'time': [0.0, 1.0], 'uv': [0.0, 0.0]})

            # 2. Parse Pump Backpressure Block with Intensity Multiplier
            has_pump = first_col.str.contains('pump', case=False, na=False)
            has_pressure = first_col.str.contains('pressure', case=False, na=False)
            has_no_degas = ~first_col.str.contains('degas', case=False, na=False)

            bp_matches = has_pump & has_pressure & has_no_degas
            if bp_matches.any():
                match_idx = bp_matches.values.argmax()
                bp_header_block = df.iloc[match_idx:match_idx + 8, :2].astype(str)

                # Extract Intensity Multiplier from block header
                mult_mask = bp_header_block.iloc[:, 0].str.contains('Intensity Multiplier', case=False, na=False)
                if mult_mask.any():
                    try:
                        multiplier = float(bp_header_block[mult_mask].iloc[0, 1])
                    except (ValueError, TypeError):
                        multiplier = 1.0
                else:
                    multiplier = 1.0

                bp_start = match_idx + 8
                has_degas_after = first_col.str.contains('degas', case=False, na=False)
                bp_end = (
                    has_degas_after.values.argmax()
                    if has_degas_after.any()
                    else len(df)
                )

                bp_df = df.iloc[bp_start:bp_end, :2].dropna().copy()
                bp_df.columns = ['time', 'pressure']
                bp_df['time'] = pd.to_numeric(bp_df['time'], errors='coerce')
                bp_df['pressure'] = pd.to_numeric(bp_df['pressure'], errors='coerce') * multiplier
                bp_df = bp_df.dropna().reset_index(drop=True)
            else:
                bp_df = pd.DataFrame({'time': [0.0, 1.0], 'pressure': [0.0, 0.0]})

            # Resolve pressure max specification bound
            max_p_spec = st.session_state.get('grad_bpspec', None)
            if max_p_spec is None or float(max_p_spec) <= 0:
                max_p_spec = float(bp_df['pressure'].max() * 1.15) if not bp_df.empty else 5000.0
            else:
                max_p_spec = float(max_p_spec)

            # 3. Deterministic %B Construction for Current Run
            curr_t_init = getattr(next_sns, 'initial_hold', 0.0)
            curr_t_g = getattr(current_sns, 'tG', 20.0)
            curr_t_final = getattr(next_sns, 'final_hold', 0.0)
            curr_c1 = getattr(current_sns, 'c1', 5.0)
            curr_c2 = getattr(current_sns, 'c2', 95.0)

            total_run_time = float(
                uv_df['time'].max()
                if not uv_df.empty
                else (curr_t_init + curr_t_g + curr_t_final + 2.0)
            )

            # Timeline Keypoints (Shifted by t_delay for detector alignment)
            t_ramp_start = curr_t_init 
            t_ramp_end = curr_t_init + curr_t_g 
            t_flush_start = curr_t_init + curr_t_g + curr_t_final 
            t_flush_end = min(total_run_time, t_flush_start + 0.5)

            curr_b_times = [
                0.0,
                t_ramp_start,
                t_ramp_end,
                t_flush_start,
                t_flush_end,
                total_run_time,
            ]
            curr_b_values = [curr_c1, curr_c1, curr_c2, curr_c2, curr_c1, curr_c1]

            # 4. Create Plot with 3 Independent Y-Axes
            fig1 = go.Figure()

            # UV Trace (Axis 1 - Left)
            fig1.add_trace(
                go.Scatter(
                    x=uv_df['time'],
                    y=uv_df['uv'],
                    name='Chromatogram Peaks (UV)',
                    line=dict(color=COLOR_BLACK, width=1.2),
                    yaxis='y',
                )
            )

            # %B Method Profile (Axis 2 - Right)
            fig1.add_trace(
                go.Scatter(
                    x=curr_b_times,
                    y=curr_b_values,
                    name='Method %B Profile',
                    line=dict(color=COLOR_NAVY, width=2.5),
                    yaxis='y2',
                )
            )

            # Backpressure Trace (Axis 3 - Far Right Offset Axis)
            fig1.add_trace(
                go.Scatter(
                    x=bp_df['time'],
                    y=bp_df['pressure'],
                    name='Pump Pressure (psi)',
                    line=dict(color=COLOR_PRESSURE, width=1.2, dash='dash'),
                    yaxis='y3',
                )
            )

            # Vertical Boundaries
            for b_t in [t_ramp_start, t_ramp_end, t_flush_start]:
                fig1.add_vline(
                    x=b_t,
                    line_width=1,
                    line_dash='dot',
                    line_color=COLOR_GRID,
                )

            # Offset Annotations
            if curr_t_init > 0:
                fig1.add_annotation(
                    x=t_ramp_start / 2.0,
                    y=curr_c1 + 12,
                    text=f'Initial Hold<br>({curr_t_init:.2f} min)',
                    showarrow=False,
                    yref='y2',
                    font=dict(size=10, color=COLOR_NAVY, family=FONT_FAMILY),
                )

            fig1.add_annotation(
                x=t_ramp_start + (curr_t_g / 2.0),
                y=(curr_c1 + curr_c2) / 2.0 + 8,
                text=f'tG Ramp<br>({curr_t_g:.2f} min)',
                showarrow=False,
                yref='y2',
                font=dict(size=11, color=COLOR_NAVY, family=FONT_FAMILY),
            )

            if curr_t_final > 0:
                fig1.add_annotation(
                    x=t_ramp_end + (curr_t_final / 2.0),
                    y=curr_c2 - 12,
                    text=f'Final Hold<br>({curr_t_final:.2f} min)',
                    showarrow=False,
                    yref='y2',
                    font=dict(size=10, color=COLOR_NAVY, family=FONT_FAMILY),
                )

            fig1.update_layout(
                title=dict(
                    text='<b>Current Run Structure: Chromatogram Peaks vs Method %B Profile</b>',
                    font=dict(size=14, color=COLOR_BLACK, family=FONT_FAMILY),
                    x=0.0,
                    y=0.98,
                ),
                xaxis=dict(
                    title=dict(
                        text='<b>Retention Time (min)</b>',
                        font=dict(size=12, color=COLOR_BLACK, family=FONT_FAMILY),
                    ),
                    tickfont=dict(size=11, color=COLOR_BLACK, family=FONT_FAMILY),
                    ticks='outside',
                    ticklen=5,
                    tickwidth=1,
                    tickcolor=COLOR_BLACK,
                    showline=True,
                    linecolor=COLOR_BLACK,
                    showgrid=False,
                    domain=[0.0, 0.85],
                ),
                # Y1: UV Signal (Left)
                yaxis=dict(
                    title=dict(
                        text='<b>UV Absorbance</b>',
                        font=dict(size=12, color=COLOR_BLACK, family=FONT_FAMILY),
                    ),
                    tickfont=dict(size=11, color=COLOR_BLACK, family=FONT_FAMILY),
                    ticks='outside',
                    ticklen=5,
                    tickwidth=1,
                    tickcolor=COLOR_BLACK,
                    showline=True,
                    linecolor=COLOR_BLACK,
                    showgrid=True,
                    gridcolor=COLOR_GRID,
                    gridwidth=1,
                ),
                # Y2: Mobile Phase %B (Right Edge of Plot Domain = 0.85)
                yaxis2=dict(
                    title=dict(
                        text='<b>Mobile Phase %B</b>',
                        font=dict(size=12, color=COLOR_NAVY, family=FONT_FAMILY),
                    ),
                    tickfont=dict(size=11, color=COLOR_NAVY, family=FONT_FAMILY),
                    range=[0, 100],
                    autorange=False,
                    ticks='outside',
                    ticklen=5,
                    tickwidth=1,
                    tickcolor=COLOR_NAVY,
                    showline=True,
                    linecolor=COLOR_NAVY,
                    overlaying='y',
                    side='right',
                    anchor='free',
                    position=0.85,
                    showgrid=False,
                ),
                # Y3: Pump Pressure (Ranging from 0 to st.session_state.grad_bpspec)
                yaxis3=dict(
                    title=dict(
                        text='<b>Pressure (psi)</b>',
                        font=dict(size=12, color=COLOR_BLACK, family=FONT_FAMILY),
                    ),
                    tickfont=dict(size=11, color=COLOR_BLACK, family=FONT_FAMILY),
                    range=[0, max_p_spec],
                    autorange=False,
                    ticks='outside',
                    ticklen=5,
                    tickwidth=1,
                    tickcolor=COLOR_BLACK,
                    showline=True,
                    linecolor=COLOR_BLACK,
                    overlaying='y',
                    side='right',
                    anchor='free',
                    position=0.98,
                    showgrid=False,
                ),
                hovermode='x unified',
                height=480,
                template='plotly_white',
                paper_bgcolor='#FFFFFF',
                plot_bgcolor='#FFFFFF',
                legend=dict(
                    title=dict(text='<b>Legend</b>', font=dict(size=10, color=COLOR_BLACK, family=FONT_FAMILY)),
                    orientation='h',
                    yanchor='bottom',
                    y=1.03,
                    xanchor='right',
                    x=1,
                    font=dict(size=10, color=COLOR_BLACK, family=FONT_FAMILY),
                ),
                margin=dict(l=50, r=60, t=100, b=50),
            )

            st.plotly_chart(fig1, use_container_width=True)

        # =====================================================================
        # TAB 2: NEXT OPTIMIZED RUN
        # =====================================================================
        with tab_next:
            if next_sns is None:
                st.info("Next run conditions not yet confirmed.")
            else:
                next_t_init = getattr(next_sns, 'initial_hold', 0.0)
                next_t_g = getattr(next_sns, 'tG', curr_t_g)
                next_t_final = getattr(next_sns, 'final_hold', 0.0)
                next_c1 = getattr(next_sns, 'c1', curr_c1)
                next_c2 = getattr(next_sns, 'c2', curr_c2)

                next_t_ramp_start = next_t_init
                next_t_ramp_end = next_t_init + next_t_g
                next_t_flush_start = next_t_init + next_t_g + next_t_final
                next_t_total = max(total_run_time, next_t_flush_start + 2.0)

                next_b_times = [
                    0.0,
                    next_t_ramp_start,
                    next_t_ramp_end,
                    next_t_flush_start,
                    next_t_flush_start + 0.5,
                    next_t_total,
                ]
                next_b_values = [
                    next_c1,
                    next_c1,
                    next_c2,
                    next_c2,
                    next_c1,
                    next_c1,
                ]

                fig2 = go.Figure()

                fig2.add_trace(
                    go.Scatter(
                        x=next_b_times,
                        y=next_b_values,
                        name='Target %B Profile',
                        line=dict(color=COLOR_NAVY, width=2.5),
                        mode='lines+markers',
                        marker=dict(size=5, color=COLOR_NAVY),
                    )
                )

                for b_t in [next_t_ramp_start, next_t_ramp_end, next_t_flush_start]:
                    fig2.add_vline(
                        x=b_t,
                        line_width=1,
                        line_dash='dot',
                        line_color=COLOR_GRID,
                    )

                if next_t_init > 0:
                    fig2.add_annotation(
                        x=next_t_ramp_start / 2.0,
                        y=next_c1 + 12,
                        text=f'Initial Hold<br>({next_t_init:.2f} min)',
                        showarrow=False,
                        font=dict(size=10, color=COLOR_NAVY, family=FONT_FAMILY),
                    )

                fig2.add_annotation(
                    x=next_t_ramp_start + (next_t_g / 2.0),
                    y=(next_c1 + next_c2) / 2.0 + 8,
                    text=f'Target tG Ramp<br>({next_t_g:.2f} min)',
                    showarrow=False,
                    font=dict(size=11, color=COLOR_NAVY, family=FONT_FAMILY),
                )

                if next_t_final > 0:
                    fig2.add_annotation(
                        x=next_t_ramp_end + (next_t_final / 2.0),
                        y=next_c2 - 12,
                        text=f'Final Hold<br>({next_t_final:.2f} min)',
                        showarrow=False,
                        font=dict(size=10, color=COLOR_NAVY, family=FONT_FAMILY),
                    )

                fig2.add_annotation(
                    x=next_t_flush_start + 0.75,
                    y=next_c1 + 15,
                    text='Flush & Re-eq',
                    showarrow=False,
                    font=dict(size=10, color=COLOR_PRESSURE, family=FONT_FAMILY),
                )

                fig2.update_layout(
                    title=dict(
                        text='<b>Next Optimized Run Structure: Target Method %B Profile</b>',
                        font=dict(size=14, color=COLOR_BLACK, family=FONT_FAMILY),
                        x=0.0,
                        y=0.98,
                    ),
                    xaxis=dict(
                        title=dict(
                            text='<b>Retention Time (min)</b>',
                            font=dict(size=12, color=COLOR_BLACK, family=FONT_FAMILY),
                        ),
                        tickfont=dict(size=11, color=COLOR_BLACK, family=FONT_FAMILY),
                        ticks='outside',
                        ticklen=5,
                        tickwidth=1,
                        tickcolor=COLOR_BLACK,
                        showline=True,
                        linecolor=COLOR_BLACK,
                        showgrid=False,
                    ),
                    yaxis=dict(
                        title=dict(
                            text='<b>Mobile Phase %B</b>',
                            font=dict(size=12, color=COLOR_BLACK, family=FONT_FAMILY),
                        ),
                        tickfont=dict(size=11, color=COLOR_BLACK, family=FONT_FAMILY),
                        range=[0, 100],
                        autorange=False,
                        ticks='outside',
                        ticklen=5,
                        tickwidth=1,
                        tickcolor=COLOR_BLACK,
                        showline=True,
                        linecolor=COLOR_BLACK,
                        showgrid=True,
                        gridcolor=COLOR_GRID,
                        gridwidth=1,
                    ),
                    hovermode='x unified',
                    height=460,
                    template='plotly_white',
                    paper_bgcolor='#FFFFFF',
                    plot_bgcolor='#FFFFFF',
                    legend=dict(
                        title=dict(text='<b>Legend</b>', font=dict(size=10, color=COLOR_BLACK, family=FONT_FAMILY)),
                        orientation='h',
                        yanchor='bottom',
                        y=1.03,
                        xanchor='right',
                        x=1,
                        font=dict(size=10, color=COLOR_BLACK, family=FONT_FAMILY),
                    ),
                    margin=dict(l=50, r=50, t=100, b=50),
                )

                st.plotly_chart(fig2, use_container_width=True)

def parse_file_name(filename: str) -> sns:
    """Parses HPLC parameter metadata from standard filename format.

    If parsing fails, prompts the user via a Streamlit form to input missing values.
    """
    # Exemplary file names:
    # GR37-34-4-MeOH-3-GRA-25-95-10-1.5-20u-CP-0.1-C18P-150x4.6x5-amb
    # GR41-23-2-MeOH-3-GRA-5-100-20-1.5-20u-CP-0.1-PFPP-100x4.6x5-amb-254

    clean_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
    split_apart = clean_name.split('-')

    try:
        if len(split_apart) < 16:
            raise IndexError("Filename does not contain enough hyphen-separated tokens.")

        # 1. Numeric Gradient & Flow Parameters
        initial_concentration = float(split_apart[6])
        final_concentration = float(split_apart[7])
        gradient_time = float(split_apart[8])
        flow_rate = float(split_apart[9])

        # 2. String Metadata (Solvent & Ligand)
        solvent = str(split_apart[3])
        ligand = str(split_apart[13])

        # 3. Column Geometry Parsing (e.g., "150x4.6x5" -> L=150.0, id=4.6)
        dim_parts = split_apart[14].split('x')
        length = float(dim_parts[0])
        internal_diameter = float(dim_parts[1])

        # 4. Temperature Handling ("amb" -> 25.0 °C or numeric string)
        temp_raw = str(split_apart[15]).lower()
        if temp_raw == 'amb':
            temp = 25.0
        else:
            try:
                temp = float(temp_raw)
            except ValueError:
                temp = 25.0  # Fallback to ambient temperature

        return sns(
            c1=initial_concentration,
            c2=final_concentration,
            tG=gradient_time,
            F=flow_rate,
            solvent=solvent,
            ligand=ligand,
            T=temp,
            id=internal_diameter,
            L=length,
            initial_hold=0.0,
            final_hold=0.0,
        )

    except (IndexError, ValueError) as err:
        st.warning(
            f"⚠️ **Invalid Filename Format**: The uploaded file name (`{filename}`) does not "
            "follow the required standard naming scheme (e.g., `GR37-34-4-MeOH-3-GRA-25-95-10-1.5-20u-CP-0.1-C18P-150x4.6x5-amb`).\n\n"
            "Please manually specify your method conditions below to proceed:"
        )

        form_key = f"manual_param_form_{hash(filename)}"
        with st.form(key=form_key):
            col1, col2 = st.columns(2)
            with col1:
                c1 = st.number_input("Initial %B (c1)", value=5.0, step=1.0)
                c2 = st.number_input("Final %B (c2)", value=95.0, step=1.0)
                tG = st.number_input("Gradient Time tG (min)", value=20.0, step=0.5)
                F = st.number_input("Flow Rate F (mL/min)", value=1.0, step=0.1)
                solvent = st.text_input("Solvent (e.g., MeOH, ACN)", value="MeOH")

            with col2:
                ligand = st.text_input("Column Stationary Phase / Ligand", value="C18")
                length = st.number_input("Column Length L (mm)", value=150.0, step=10.0)
                internal_diameter = st.number_input("Column Internal Diameter id (mm)", value=4.6, step=0.1)
                T = st.number_input("Temperature T (°C)", value=25.0, step=1.0)

            submitted = st.form_submit_button("Confirm Method Conditions")

        if submitted:
            return sns(
                c1=c1,
                c2=c2,
                tG=tG,
                F=F,
                solvent=solvent,
                ligand=ligand,
                T=T,
                id=internal_diameter,
                L=length,
                initial_hold=0.0,
                final_hold=0.0,
            )
        else:
            st.stop()

def find_backpressure(df: pd.DataFrame) -> sns:
    first_col = df.iloc[:, 0].astype(str)
    
    # 1. Define individual boolean masks
    has_pump = first_col.str.contains('pump', case=False, na=False)
    has_pressure = first_col.str.contains('pressure', case=False, na=False)
    has_no_degas = ~first_col.str.contains('degas', case=False, na=False)
    has_degas = first_col.str.contains('degas', case=False, na=False)
    
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
    end_row_match = has_pressure & has_degas
    end_row = end_row_match.values.argmax() if end_row_match.any() else len(df)
    
    # Extract search area and explicitly convert values to numeric float
    raw_search = df.iloc[start_row:end_row, 1]
    numeric_search = pd.to_numeric(raw_search, errors='coerce').dropna()
    
    if numeric_search.empty:
        backpressure = 0.0
    else:
        max_in_search = float(numeric_search.max())
        backpressure = max_in_search * float(multiplier)
        
    return sns(bp=backpressure)


def determine_holds_from_bp(
    df: pd.DataFrame, F: float, L: float, id_mm: float, tG: float
) -> sns:
    """Detects initial hold and final high-%B hold using relative pressure baseline

    deflection and backward regression snakes.
    """
    first_col = df.iloc[:, 0].astype(str)

    # 1. Locate backpressure data block
    has_pump = first_col.str.contains('pump', case=False, na=False)
    has_pressure = first_col.str.contains('pressure', case=False, na=False)
    has_no_degas = ~first_col.str.contains('degas', case=False, na=False)

    matches = has_pump & has_pressure & has_no_degas
    if not matches.any():
        return sns(initial_hold=0.0, final_hold=0.0)

    row_of_match = matches.values.argmax()
    start_row = row_of_match + 8

    has_degas = first_col.str.contains('degas', case=False, na=False)
    end_row_mask = has_pressure & has_degas
    end_row = end_row_mask.values.argmax() if end_row_mask.any() else len(df)

    # Extract retention time (col 0) and backpressure intensity (col 1)
    bp_data = df.iloc[start_row:end_row, :2].dropna().copy()
    bp_data.columns = ['time', 'intensity']
    bp_data['time'] = pd.to_numeric(bp_data['time'], errors='coerce')
    bp_data['intensity'] = pd.to_numeric(bp_data['intensity'], errors='coerce')
    bp_data = bp_data.dropna().reset_index(drop=True)

    if len(bp_data) < 30:
        return sns(initial_hold=0.0, final_hold=0.0)

    # Subsample every 3rd point (step size = 3)
    subsampled = bp_data.iloc[::3].reset_index(drop=True)

    # Calculate column void volume delay (V_m / F)
    radius_cm = (id_mm / 10.0) / 2.0
    length_cm = L / 10.0
    v_m = np.pi * (radius_cm**2) * length_cm * 0.68  # mL
    t_delay = v_m / F if F > 0 else 0.0

    window_size = 10

    # Baseline pressure parameters for relative deflection check
    p_baseline = float(subsampled['intensity'].iloc[:10].median())
    p_max = float(subsampled['intensity'].max())
    p_range = max(1.0, p_max - p_baseline)

    # ------------------------------------------------------------------
    # 2. INITIAL HOLD SNAKE (Forward Walk)
    # Detects start of main gradient ramp relative to initial baseline
    # ------------------------------------------------------------------
    t_ramp_start = None

    for i in range(len(subsampled) - window_size + 1):
        window = subsampled.iloc[i : i + window_size]
        x = window['time'].values
        y = window['intensity'].values

        slope, _, r_val, _, _ = linregress(x, y)
        r_squared = r_val**2

        # Deflection criteria: High linear fit AND pressure rising > 2% of total range
        window_avg_p = np.mean(y)
        p_deflection = window_avg_p - p_baseline

        if r_squared >= 0.92 and slope > 0 and p_deflection >= (0.02 * p_range):
            t_ramp_start = float(window['time'].iloc[0])
            break

    if t_ramp_start is None:
        print('t ramp start is None!')
        t_ramp_start = 0.0

    raw_initial = max(0.0, t_ramp_start - t_delay)
    initial_hold = round(raw_initial * 4.0) / 4.0

    # ------------------------------------------------------------------
    # 3. FINAL HOLD SNAKE (Backward Walk)
    # Detects start of post-hold re-equilibration flush AFTER gradient ends at detector
    # ------------------------------------------------------------------
    gradient_end_at_detector = t_ramp_start + tG + t_delay
    t_flush_start = None

    for i in range(len(subsampled) - window_size, -1, -1):
        window = subsampled.iloc[i : i + window_size]
        x = window['time'].values
        y = window['intensity'].values

        t_window_start = float(window['time'].iloc[0])

        # Ignore windows occurring BEFORE the gradient completes at detector
        if t_window_start < gradient_end_at_detector:
            continue

        slope, _, r_val, _, _ = linregress(x, y)
        r_squared = r_val**2

        # Negative slope drop-off exceeding 2% range per minute
        if r_squared >= 0.90 and slope < -(0.02 * p_range):
            t_flush_start = t_window_start
            break  # Stop at the FIRST valid re-equilibration drop coming from the end

    # Calculate final hold from detected flush transition
    if t_flush_start is not None:
        raw_final = max(0.0, t_flush_start - gradient_end_at_detector)
        final_hold = round(raw_final * 4.0) / 4.0
        print(f'raw_final {raw_final}. gradient_end_at_detector = {gradient_end_at_detector}. t_flush_start {t_flush_start}')
    else:
        print('T Flush Start is NONE!')
        # Dynamic fallback: remaining non-gradient window after initial hold
        total_time = float(bp_data['time'].max())
        remaining = max(0.0, total_time - tG - initial_hold)
        final_hold = round(remaining * 4.0) / 4.0

    return sns(initial_hold=initial_hold, final_hold=final_hold)

def run_structure(df: pd.DataFrame, filename: str) -> sns:
    """Determines run structure by using independent deflection searches for

    Initial and Final Isocratic Holds, maintaining a dedicated pipeline for total
    non-gradient window duration.
    """
    # 1. Parse metadata from filename
    parsed_sns = parse_file_name(filename)

    # 2. Extract total run time from chromatogram block
    first_col = df.iloc[:, 0].astype(str)
    has_chrom = first_col.str.contains('chromatogram', case=False, na=False)
    has_det = first_col.str.contains('detector', case=False, na=False)
    has_no_press = ~first_col.str.contains('pressure', case=False, na=False)
    has_no_temp = ~first_col.str.contains('temp', case=False, na=False)

    target_matches = has_chrom & has_det & has_no_press & has_no_temp
    if not target_matches.any():
        raise ValueError(
            'Could not locate chromatogram detector block in file header.'
        )

    row_of_match = target_matches.values.argmax()
    start_data_row = row_of_match + 8

    has_pressure = first_col.str.contains('pressure', case=False, na=False)
    end_matches = has_pressure & (first_col.index > start_data_row)
    end_data_row = end_matches.values.argmax() if end_matches.any() else len(df)

    raw_time_col = df.iloc[start_data_row:end_data_row, 0]
    numeric_time = pd.to_numeric(raw_time_col, errors='coerce').dropna()
    total_time = (
        float(numeric_time.max()) if not numeric_time.empty else parsed_sns.tG
    )

    # --- SEPARATE PIPE: Total Non-Gradient Window ---
    total_non_grad_window = max(0.0, total_time - parsed_sns.tG)

    # 3. Detect initial and final holds independently from backpressure profile
    detected_holds = determine_holds_from_bp(
        df=df,
        F=parsed_sns.F,
        L=parsed_sns.L,
        id_mm=parsed_sns.id,
        tG=parsed_sns.tG,
    )

    raw_init_hold = min(detected_holds.initial_hold, total_non_grad_window)
    raw_final_hold = min(
        detected_holds.final_hold,
        max(0.0, total_non_grad_window - raw_init_hold),
    )

    # 4. Session state management for initial hold adjustment
    init_key = f'grad_init_hold_{filename}'
    submitted_key = f'grad_hold_submitted_{filename}'

    if init_key not in st.session_state:
        st.session_state[init_key] = float(raw_init_hold)
    if submitted_key not in st.session_state:
        st.session_state[submitted_key] = False

    current_init_hold = st.session_state[init_key]
    # final_hold comes directly from the detected final hold value
    current_final_hold = raw_final_hold

    # 5. UI Display
    st.write('---')
    st.markdown('### ⏱️ **Gradient Run Profile Verification**')

    # Total non-gradient window pipe
    st.info(
        f'**Run Metrics:** Total Time = **{total_time:.2f} min** | '
        f'Gradient Time ($t_G$) = **{parsed_sns.tG:.2f} min** | '
        f'**Total Non-Gradient Window:** **{total_non_grad_window:.2f} min**'
    )

    # Primary hold values readout (both preserved as independent values)
    st.markdown(
        f'**Initial Isocratic Hold:** `{current_init_hold:.2f} min` | '
        f'**Final Isocratic Hold (High %B):** `{current_final_hold:.2f} min`'
    )

    # Interactive Adjustments for Initial Hold
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button('➖ 0.25 min', key=f'dec_{filename}'):
            st.session_state[init_key] = max(
                0.0, st.session_state[init_key] - 0.25
            )
            st.rerun()

    with col2:
        if st.button('➕ 0.25 min', key=f'inc_{filename}'):
            max_allowed = max(0.0, total_non_grad_window - current_final_hold)
            st.session_state[init_key] = min(
                max_allowed, st.session_state[init_key] + 0.25
            )
            st.rerun()

    with col3:
        if st.button(
            '✅ Confirm Isocratic Holds', type='primary', key=f'sub_{filename}'
        ):
            st.session_state[submitted_key] = True
            st.success(
                f'Locked Profile: Initial Hold = {current_init_hold:.2f} min | '
                f'Final Hold = {current_final_hold:.2f} min | '
                f'Total Non-Gradient Window = {total_non_grad_window:.2f} min.'
            )

    # Pause downstream decision engine execution until user submission
    if not st.session_state[submitted_key]:
        st.warning(
            "Please verify hold times and click 'Confirm Isocratic Holds' to"
            ' proceed.'
        )
        st.stop()

    # 6. Package and return verified parameters
    parsed_sns.total_time = total_time
    parsed_sns.total_non_grad_window = total_non_grad_window
    parsed_sns.initial_hold = current_init_hold
    parsed_sns.final_hold = current_final_hold
    

    return parsed_sns



def evaluate_next_run(df: pd.DataFrame, filename: str) -> None:
    peaks = peak_count(df)
    backpressure = find_backpressure(df).bp
    run_params = run_structure(df, filename)
    
    # Runtime decides if we finished or not the run.
    runtime = run_params.total_time
    resolution = find_resolution(df, peaks)['min_res']

    static_values = [
        'c1',
        'c2',
        'F',
        'T',
        'tG',
        'solvent',
        'ligand',
        'L'
    ]
    
    peakspec = st.session_state.grad_peakspec
    res_spec = st.session_state.grad_res_spec
    runspec = st.session_state.grad_runspec
    
    final_sns = sns()
    scouting_sns = st.session_state.grad_conditions[0]
    last_sns = st.session_state.grad_conditions[-1]
    information_string = ''

    if peaks < peakspec:
        if st.session_state.g1_progression > 2:
            # Selectivity change, since we already did increase and decrease.
            if st.session_state.selectivity_progression == 0:
                final_sns.T = 40
                information_string = (
                    f'Case G1: Temperature changed from {scouting_sns.T} °C to {final_sns.T} °C. Selectivity must change because there are not enough peaks.\n'
                    'Attempted changes to GS failed to provide sufficient peaks.'
                )
                static_values.remove('T')
                st.session_state.selectivity_progression += 1
            elif st.session_state.selectivity_progression == 1:
                final_sns.T = 60
                information_string = (
                    f'Case G1: Temperature changed from {scouting_sns.T} °C to {final_sns.T} °C. Selectivity must change because there are not enough peaks.\n'
                    'Attempted changes to GS failed to provide sufficient peaks.'
                )
                static_values.remove('T')
                st.session_state.selectivity_progression += 1
            elif st.session_state.selectivity_progression == 2:
                final_sns.solvent = 'MeOH' if scouting_sns.solvent.strip().lower() == 'acn' else 'ACN'
                information_string = (
                    f'Case G1: Solvent changed from {scouting_sns.solvent} to {final_sns.solvent}. Selectivity must change because there are not enough peaks.\n'
                    'Attempted changes to GS failed to provide sufficient peaks.'
                )
                static_values.remove('solvent')
                st.session_state.selectivity_progression += 1
            elif st.session_state.selectivity_progression == 3:
                final_sns.T = 40
                information_string = (
                    f'Case G1: Temperature changed from {scouting_sns.T} °C to {final_sns.T} °C. Selectivity must change because there are not enough peaks.\n'
                    'Attempted changes to GS failed to provide sufficient peaks.'
                )
                static_values.remove('T')
                st.session_state.selectivity_progression = 1

        elif (st.session_state.g1_progression % 2) == 0:
            what_happens = 'Decrease'
            final_sns.tG = 0.5 * scouting_sns.tG
            information_string = (
                f'Case G1: {what_happens} tG from the scouting {scouting_sns.tG} min to {final_sns.tG} min.\n'
                'Selectivity must change because there are not enough peaks. Changing Gradient Steepness (GS) is required.'
            )
            st.session_state.g1_progression += 1
            static_values.remove('tG')
        else:
            what_happens = 'Increase'
            final_sns.tG = 1.2 * scouting_sns.tG
            information_string = (
                f'Case G1: {what_happens} tG from the scouting {scouting_sns.tG} min to {final_sns.tG} min.\n'
                'Selectivity must change because there are not enough peaks. Changing Gradient Steepness (GS) is required.'
            )
            st.session_state.g1_progression += 1
            static_values.remove('tG')

    elif peaks == peakspec:
        if resolution < res_spec:
            # Stretch the run: Reduce flow rate (F * 0.8), increase tG (tG / 0.8)
            final_sns.tG = last_sns.tG / 0.8
            final_sns.F = last_sns.F * 0.8
            information_string = (
                f'Case G2: Increase tG from last run\'s {last_sns.tG:.2f} min to {final_sns.tG:.2f} min '
                f'and decrease flow rate from {last_sns.F:.2f} mL/min to {final_sns.F:.2f} mL/min to improve resolution.'
            )
            static_values.remove('tG')
            static_values.remove('F')

        elif resolution >= res_spec and runtime <= runspec:
            information_string = 'Case G4: Run is complete! Lock it.'

        elif resolution >= res_spec and runtime > runspec:
            # --- Case G3: Shrink Gradient Window while keeping Gradient Steepness (GS) constant ---
            ret_data = retention_time(df, peaks)
            rts = ret_data['retention_times']

            # 1. Retention times of first analyte peak (index 1, skipping t0 at index 0) and last peak
            tR_first = float(rts[1]) if len(rts) > 1 else float(rts[0])
            tR_last = float(rts[-1])

            # 2. Gradient slope (%B / min)
            c1_old = last_sns.c1
            c2_old = last_sns.c2
            tG_old = last_sns.tG
            slope = (c2_old - c1_old) / tG_old if tG_old > 0 else 1.0

            # 3. Calculate %B at elution subtracting initial isocratic hold
            t_hold_init = run_params.initial_hold
            
            t_grad_first = max(0.0, tR_first - t_hold_init)
            t_grad_last = max(0.0, tR_last - t_hold_init)

            b_first = c1_old + (t_grad_first * slope)
            b_last = c1_old + (t_grad_last * slope)

            # 4. Apply safety boundary floors (Max 35% starting %B, Min 70% ending %B)
            c1_new = min(35.0, b_first)
            c2_new = max(70.0, b_last)

            # Round to clean 1 decimal place
            final_sns.c1 = round(c1_new, 1)
            final_sns.c2 = round(c2_new, 1)

            # Recalculate tG to maintain exact same gradient slope (GS)
            final_sns.tG = round((final_sns.c2 - final_sns.c1) / slope, 2)

            information_string = (
                f'Case G3: Shrinking gradient window from {c1_old:.1f}%–{c2_old:.1f}% B to '
                f'{final_sns.c1:.1f}%–{final_sns.c2:.1f}% B.\n'
                f'First peak elutes at {b_first:.1f}% B ({tR_first:.2f} min) and last peak elutes at {b_last:.1f}% B ({tR_last:.2f} min).\n'
                f'Adjusted tG from {tG_old:.2f} min to {final_sns.tG:.2f} min to maintain constant gradient steepness.'
            )

            static_values.remove('c1')
            static_values.remove('c2')
            static_values.remove('tG')


    # Copy static values from last_sns that were not updated dynamically
    for key in static_values:
        setattr(final_sns, key, getattr(last_sns, key))

    final_sns.initial_hold = run_params.initial_hold
    final_sns.final_hold = run_params.final_hold
    
    # --- Calculate Column Void Volume (V_m = pi * r^2 * L * 0.68) ---
    # Convert i.d. (mm) and L (mm) to cm
    radius_cm = (getattr(final_sns, 'id', 4.6) / 10.0) / 2.0
    length_cm = getattr(final_sns, 'L', 150.0) / 10.0
    v_m = np.pi * (radius_cm ** 2) * length_cm * 0.68  # Volume in mL

    # --- Calculate Complete Dimensionless Gradient Steepness (GS = (DeltaPhi * Vm) / (tG * F)) ---
    # Previous Run GS
    delta_phi_prev = (last_sns.c2 - last_sns.c1) / 100.0
    denom_prev = last_sns.tG * last_sns.F
    gs_prev = (delta_phi_prev * v_m) / denom_prev if denom_prev > 0 else 0.0

    # New Run GS
    delta_phi_new = (final_sns.c2 - final_sns.c1) / 100.0
    denom_new = final_sns.tG * final_sns.F
    gs_new = (delta_phi_new * v_m) / denom_new if denom_new > 0 else 0.0

    # Tolerance check for floating point comparison
    gs_changed = abs(gs_new - gs_prev) > 1e-4

    if gs_changed:
        gs_summary = f"\n📊 **Gradient Steepness (GS):** Changed from **{gs_prev:.4f}** to **{gs_new:.4f}**."
    else:
        gs_summary = f"\n📊 **Gradient Steepness (GS):** Remained unchanged at **{gs_prev:.4f}**."

    # Append GS evaluation summary to information_string
    information_string += gs_summary

    # Append new condition profile to session state
    st.session_state.grad_expected_run.append(final_sns)
    st.info(information_string)


def main() -> None:
    initialise_variables()
    with st.expander('View Instructions and Video'):
        pass
    with st.container():
        st.header('Method Specifications')
        st.caption('Edit then submit these specifications.')
    
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
    if 'spec_submitted' not in st.session_state:
        st.session_state['spec_submitted'] = False
    if st.button('Submit Specifications'):
        st.session_state.grad_runspec = runspec
        st.session_state.grad_peakspec = peakspec
        st.session_state.grad_res_spec = res_spec
        st.session_state.grad_bpspec = bpspec
        st.session_state.spec_submitted = True

    added_file = st.file_uploader('Upload a CSV file of your run.', type='csv', key="main_run_uploader")
    if added_file is not None:
        file_id = f"{added_file.name}_{added_file.size}"
        if not st.session_state.spec_submitted:
            st.session_state.grad_runspec = runspec
            st.session_state.grad_peakspec = peakspec
            st.session_state.grad_res_spec = res_spec
            st.session_state.grad_bpspec = bpspec
            st.session_state.spec_submitted = True
        if file_id not in st.session_state.grad_files:
            run_conditions = parse_file_name(added_file.name)
            df = safe_read(added_file)
            if st.session_state.grad_files is None or st.session_state.grad_expected_run == []:
                st.session_state.grad_files = []
                st.session_state['grad_conditions'] = []
                st.session_state.grad_conditions.append(run_conditions)
                evaluate_next_run(df, added_file.name)
            else:
                expected = st.session_state.grad_expected_run[-1]
                # Method Deviation Check
                if run_conditions != expected:
                    # Report deviation
                    st.warning(
                        f'Method Deviation Observed.\n'
                        f'**Expected tG**: {expected.tG} min. **Actual tG**: {run_conditions.tG} min.\n'
                        f'**Expected Temperature**: {expected.T} ºC. **Actual Temperature**: {run_conditions.T} ºC.\n'
                        f'**Expected 𝚫ɸ**: {(expected.c2 - expected.c1)/100.0} ({expected.c1}% to {expected.c2}%)'
                        f'**Actual 𝚫ɸ**: {(run_conditions.c2 - run_conditions.c1)/100.0} ({run_conditions.c1}% to {run_conditions.c2}%)\n'
                        f'**Expected Solvent**: {expected.solvent}. Actual Solvent: {run_conditions.solvent}'
                    )
                    st.session_state.grad_conditions.append(run_conditions)
                else:
                    st.session_state.grad_conditions.append(expected)
                evaluate_next_run(df, added_file.name)
                
            st.session_state.grad_files.append(file_id)
            pass
            display_gradient_run_structure_expander(df, st.session_state.grad_conditions[-1], st.session_state.grad_expected_run[-1])

main()
