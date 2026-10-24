import sys
from pathlib import Path

# Add project root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from Engine import safe_read, peak_count, find_backpressure, find_resolution, retention_time
import streamlit as st
from types import SimpleNamespace as sns

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
    static_keys = [
        'grad_runspec',
        'grad_bpspec'
        'grad_res_spec'
        'grad_peakspec',
        'grad_files',
        'grad_F',
        'grad_c1',
        'grad_c2',
        'grad_tG',
        'grad_solvent',
        'grad_ligand',
        'grad_temperature'
    ]
    for each in static_keys:
        if each not in st.session_state:
            st.session_state[each] = None
    
    return None


def parse_file_name(filename: str) -> sns:
    # Exemplary file names:
    # GR37-34-4-MeOH-3-GRA-25-95-10-1.5-20u-CP-0.1-C18P-150x4.6x5-amb
    # GR41-23-2-MeOH-3-GRA-5-100-20-1.5-20u-CP-0.1-PFPP-100x4.6x5-amb-254
    # Format: Group Number, Injection #, Solvent, pH, Method, C1, C2, tG, F, Inj Vol, Type of Injection, Sample Concentration, Column
    split_apart = filename.split('-')
    initial_concentration = float(split_apart[6])
    final_concentration = float(split_apart[7])
    gradient_time = float(split_apart[8])
    flow_rate = float(split_apart[9])
    temp = float(split_apart[15])
    ligand = float(split_apart[13])
    solvent = float(split_apart[3])
    return sns(
        c1=initial_concentration,
        c2=final_concentration,
        tG=gradient_time,
        F=flow_rate,
        solvent=solvent,
        ligand=ligand,
        temp=temp
    )

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

    if st.button('Submit Specifications'):
        st.session_state.grad_runspec = runspec
        st.session_state.grad_peakspec = peakspec
        st.session_state.grad_res_spec = res_spec
        st.session_state.grad_bpspec = bpspec

    added_file = st.file_uploader('Upload a CSV file of your run.', type='csv', key="main_run_uploader")
    if added_file is not None:
        file_id = f"{added_file.name}_{added_file.size}"
        if file_id not in st.session_state.grad_files:
            if st.session_state.grad_files is None:
                st.session_state.grad_files = []
                run_conditions = parse_file_name(added_file.name)

            st.session_state.grad_files.append(file_id)
            pass

main()
