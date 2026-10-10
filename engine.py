

# Necessary imports
import sys
from streamlit.web import cli as stcli
import streamlit as st
import pandas as pd
import chardet
import numpy as np
import io

def safe_read(uploaded_file):
    # A way to read the file without risk of encoding errors.
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
            
        resolutions.append(parsed_val)
    
    # Correct list comprehension syntax: [x for x in resolutions if ...]
    non_zero_resolutions = [x for x in resolutions if x > 0.0]
    
    output = {
        "resolutions": resolutions,
        "min_res": float(min(resolutions)) if resolutions else 0.0,
        # Uses default parameter to prevent crash if no non-zero resolution exists
        "min_res_no_zero": float(min(non_zero_resolutions, default=0.0))
    }
    
    return output

def main():
    st.set_page_config(layout='wide')
    # Title & instructions
    st.title("Ahmed's HPLC Method Development Engine in Python")
    st.text('Edit the specifications and run conditions to match your method. An example is provided. Hover over the (?) icon for information.')
    
    leftside, rightside = st.columns(2)
    # Left side of page: not column.
    with leftside:
        # Border wrapping
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

    # What concerns the column
    with rightside:
        # Border wrapping
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
        'all_sample_concentrations'
    ]

    for key in default_keys:
        if key not in st.session_state:
            st.session_state[key] = []

    # Tracker set to prevent re-parsing on every Streamlit page re-render
    if 'processed_file_ids' not in st.session_state:
        st.session_state.processed_file_ids = set()


    # 2. Uploading & Single-Execution Processing
    added_file = st.file_uploader('Upload a CSV file of your run.', type='csv')

    if added_file is not None:
        # Unique file fingerprint combining filename and byte size
        file_id = f"{added_file.name}_{added_file.size}"
        
        # Process ONLY if this specific upload hasn't been parsed yet
        if file_id not in st.session_state.processed_file_ids:
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
                
                # Append parsed values to session state
                st.session_state.all_csvs.append(dataframe_added)
                st.session_state.all_peak_counts.append(peak_cnt)
                st.session_state.all_retention_times.append(retention_analysis['retention_times'])
                st.session_state.all_run_times.append(retention_analysis['runtime'])
                st.session_state.all_resolutions_overall.append(resolution_analysis['resolutions'])
                st.session_state.all_resolutions_final.append(resolution_analysis['min_res'])
                st.session_state.all_equilibrated.append(bp_analysis['equilibrated'])
                st.session_state.all_backpressures.append(bp_analysis['backpressure'])
                
                # Mark file as processed
                st.session_state.processed_file_ids.add(file_id)
                st.success(f"Successfully processed {added_file.name}!")
                
            except Exception as e:
                st.error(f"Error parsing {added_file.name}: {e}")

with st.expander("🔍 View Raw Parsed Values (Session State)", expanded=False):
        if len(st.session_state.all_csvs) == 0:
            st.info("No files have been parsed yet. Upload a CSV to view extracted data.")
        else:
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
                "backpressures": st.session_state.all_backpressures
            })

  if __name__ == "__main__":
    if not st.runtime.exists():
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())

    main()
