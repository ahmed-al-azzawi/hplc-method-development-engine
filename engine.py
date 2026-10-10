

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
    # Safe default search bounds without needing passed arguments
    b_min = float(st.session_state.all_pct_bs[-1])
    b_max = 80.0
    step = 0.5

    ret_times = list(st.session_state.all_retention_times)
    pct_bs = list(st.session_state.all_pct_bs)
    
    if not ret_times or len(ret_times) < 2:
        raise ValueError("At least two runs are required to perform linear regression.")
    
    max_peaks = max(len(sublist) - 1 for sublist in ret_times)
    curves = []
    
    # 1. Fit linear regression models
    for peak_idx in range(1, max_peaks + 1):
        xvals, yvals = [], []
        for run_idx, run_ret_times in enumerate(ret_times):
            if peak_idx < len(run_ret_times):
                t0, tR = run_ret_times[0], run_ret_times[peak_idx]
                if tR > t0 and t0 > 0:
                    k = (tR - t0) / t0
                    xvals.append(pct_bs[run_idx])
                    yvals.append(np.log10(k))
        
        if len(xvals) >= 2:
            res = linregress(xvals, yvals)
            curves.append((res.slope, res.intercept))
            
    if len(curves) < 2:
        return float(ceil(pct_bs[-1]))
        
    # Cap upper search bound to max %B run so far (or b_max default)
    actual_b_max = min(b_max, max(pct_bs))
    speculative_bs = np.arange(b_min, actual_b_max + step, step)
    
    slopes = np.array([c[0] for c in curves])[:, np.newaxis]      # Shape: (N_peaks, 1)
    intercepts = np.array([c[1] for c in curves])[:, np.newaxis]  # Shape: (N_peaks, 1)
    
    # 2. Vectorized log(k) and real k conversion: k = 10^(m * %B + c)
    log_k_grid = slopes * speculative_bs + intercepts
    k_grid = 10 ** log_k_grid  # Convert back to real k values!
    
    # 3. Vectorized pairwise difference across all peaks for every %B step
    pairwise_diffs = np.abs(k_grid[:, np.newaxis, :] - k_grid[np.newaxis, :, :])
    
    # Mask out diagonal (self-comparisons)
    n_peaks = len(curves)
    diag_mask = ~np.eye(n_peaks, dtype=bool)
    
    # Minimum separation across peaks for each speculative %B
    min_separations = pairwise_diffs[diag_mask, :].reshape(n_peaks - 1, n_peaks, -1).min(axis=(0, 1))
    
    # 4. Find %B that maximizes the worst-case real k separation
    best_idx = np.argmax(min_separations)
    ideal_pct_b = speculative_bs[best_idx]
    return float(round(ideal_pct_b, 1))


def handle_selectivity(prev_b=False):
    # 1. Initialize counter
    if 'selectivity_progression' not in st.session_state:
        st.session_state.selectivity_progression = 0
    
    st.session_state.selectivity_progression += 1
    
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
    if prev_b:
        modified_parameters.append('all_pct_bs')
        st.session_state.all_pct_bs.append(st.session_state.all_pct_bs[-2])
        
    match st.session_state.selectivity_progression:
        case 1:
            st.session_state.all_temperatures.append('40')
            modified_parameters.append('all_temperatures')
            st.info(
                "**Next Run Strategy:** Increase Temperature to 40 °C to modify band spacing (selectivity) and reduce mobile phase viscosity. Shortens runtime and may reduce peak width. \n\n"
                "*Controlled by:* Column Oven | *Category:* Both (Chemical & Mechanical) | *Cycle:* Selectivity"
            )
            
        case 2:
            st.session_state.all_temperatures.append('60')
            modified_parameters.append('all_temperatures')
            st.info(
                "**Next Run Strategy:** Increase Temperature to 60 °C to further adjust selectivity and sharpen peak shapes. Further shortens runtime and may reduce peak width. \n\n"
                "*Controlled by:* Column Oven | *Category:* Both (Chemical & Mechanical) | *Cycle:* Selectivity"
            )
            
        case 3:
            st.session_state.all_temperatures.append('amb')
            current_solvent = st.session_state.all_solvents[-1].strip().lower()
            current_pct_b = st.session_state.all_pct_bs[-1]
            
            if current_solvent in ['acn', 'acetonitrile']:
                new_solvent = 'MeOH'
                nomogram_slope = 1.0090909
                nomogram_int = 6.9264069
                converted_b = current_pct_b * nomogram_slope + nomogram_int
            elif current_solvent in ['meoh', 'methanol']:
                new_solvent = 'ACN'
                nomogram_slope = 0.980221359
                nomogram_int = -6.246035141
                converted_b = current_pct_b * nomogram_slope + nomogram_int
            else:
                new_solvent = st.session_state.all_solvents[-1]
                converted_b = current_pct_b

            st.session_state.all_solvents.append(new_solvent)
            st.session_state.all_pct_bs.append(round(converted_b, 1))
            modified_parameters.extend(['all_temperatures', 'all_pct_bs', 'all_solvents'])
            
            st.info(
                f"**Next Run Strategy:** Swap Organic Solvent to {new_solvent} at nomogram-converted ratio ({round(converted_b, 1)}% B) at ambient temperature to alter solute-solvent solvophobic interactions like acidity, basicity and dipolarity at the same strength.\n\n"
                f"*Controlled by:* Pump / Solvent Manager | *Category:* Chemical | *Cycle:* Selectivity"
            )
            
        case 4:
            st.session_state.all_temperatures.append('40')
            modified_parameters.append('all_temperatures')
            st.info(
                "**Next Run Strategy:** Increase Temperature to 40 °C to modify band spacing (selectivity) and reduce mobile phase viscosity. Shortens runtime and may reduce peak width. \n\n"
                "*Controlled by:* Column Oven | *Category:* Both (Chemical & Mechanical) | *Cycle:* Selectivity"
            )
            
        case 5:
            st.session_state.all_temperatures.append('60')
            modified_parameters.append('all_temperatures')
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
                'all_flow_rates', 'all_temperatures', 'all_sample_concentrations'
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
    # Processing of decisions
    if st.button('Click to acquire conditions of next run, according to this run.'):
        # Guard check: Ensure at least one run has been uploaded and parsed
        if not st.session_state.all_backpressures:
            st.error("No run data found. Please upload a CSV run file first.")
        else:
            latest_bp = st.session_state.all_backpressures[-1]
            latest_peaks = st.session_state.all_peak_counts[-1]
            latest_runtime = st.session_state.all_run_times[-1]
            latest_resolution = st.session_state.all_resolutions_final[-1]
            run_count = len(st.session_state.all_backpressures)
            
            # 1. Backpressure Check
            if latest_bp > bpspec:
                st.caption(f"📊 **Run Metrics:** Backpressure ({latest_bp:.1f} bar) > Spec ({bpspec:.1f} bar) | Peaks: {latest_peaks}/{peakspec}")
                st.warning(f'Run number {run_count} exceeded backpressure limit!')
                st.session_state.show_decision_radio = False
                
            # 2. Peak Count Checks
            elif latest_peaks > peakspec:
                st.caption(f"📊 **Run Metrics:** Peaks ({latest_peaks}) > Spec ({peakspec}) | $t_R$: {latest_runtime:.1f} min | $R_s$: {latest_resolution:.2f}")
                st.warning(f'Run number {run_count} has more peaks than expected. Investigate for contamination.')
                st.session_state.show_decision_radio = False
                
            elif latest_peaks < peakspec:
                # 3a. Fast Run / Fixed Step
                if latest_runtime < (0.5 * runspec):
                    st.caption(f"📊 **Run Metrics:** Peaks ({latest_peaks}) < Spec ({peakspec}) | $t_R$ ({latest_runtime:.1f} min) << Spec ({runspec} min) | $R_s$: {latest_resolution:.2f}")
                    current_pct_b = st.session_state.all_pct_bs[-1]
                    new_pct_b = current_pct_b - 10.0
                    
                    st.info(
                        f"**Next Run Strategy:** Decrease %B by 10% (from {current_pct_b}% to {new_pct_b}%)\n\n"
                        f"*Controlled by:* Pump | *Cycle:* Retention"
                    )
                    
                    st.session_state.all_pct_bs.append(new_pct_b)
                    
                    static_keys = [
                        'all_solvents', 
                        'all_ligands', 
                        'all_bead_types',
                        'all_pore_sizes', 
                        'all_coreshells', 
                        'all_carbon_loads',
                        'all_column_lengths',      # Added missing key
                        'all_internal_diameters', 
                        'all_particle_sizes',
                        'all_flow_rates', 
                        'all_temperatures'
                    ]

                    # Propagate static chromatographic parameters to the next run
                    for key in static_keys:
                        st.session_state[key].append(st.session_state[key][-1])
                    st.session_state.show_decision_radio = False

                # 3b. Optimization Decision Trigger
                elif latest_runtime < runspec:
                    st.caption(f"📊 **Run Metrics:** Peaks ({latest_peaks}) < Spec ({peakspec}) | $t_R$ ({latest_runtime:.1f} min) < Spec ({runspec} min) | $R_s$: {latest_resolution:.2f}")
                    st.session_state.choice1 = min_b(runspec)
                    st.session_state.choice2 = int_pct_b()
                    st.session_state.show_decision_radio = True
                
                elif latest_runtime > runspec:
                    st.caption(f"📊 **Run Metrics:** Peaks ({latest_peaks}) < Spec ({peakspec}) | $t_R$ ({latest_runtime:.1f} min) > Spec ({runspec} min) | $R_s$: {latest_resolution:.2f}")
                    # Where runtime above spec, and less than spec peaks, return to previous %B and change selectivity.
                    handle_selectivity(True)
            
            elif latest_peaks == peakspec:
                # 1. Define criteria thresholds
                pass_runtime = latest_runtime <= runspec
                runtime_far_below = latest_runtime < (0.5 * runspec)  # tR << Spec
                pass_resolution = latest_resolution >= res_spec

                # ------------------------------------------------------------------
                # CASE A: All specifications MET (Rs >= spec AND tR <= spec)
                # ------------------------------------------------------------------
                if pass_runtime and pass_resolution:
                    if runtime_far_below:
                        st.caption(f"📊 **Run Metrics:** Peaks ({latest_peaks}) == Spec ({peakspec}) | $t_R$ ({latest_runtime:.1f} min) << Spec ({runspec} min) | $R_s$ ({latest_resolution:.2f}) >= Spec ({res_spec})")
                        st.info(
                            f"**Next Run Strategy:** Method MET. %B and Selectivity parameters are frozen.\n\n"
                            f"*Observation:* Run time ({latest_runtime:.1f} min) is far below target (≤{runspec} min).\n\n"
                            f"*Optional Recommendation:* Move to Efficiency Cycle (Flow Chart 2) to increase flow rate and shorten run time further.\n\n"
                            f"*Controlled by:* Pump | *Category:* Mechanical | *Cycle:* Efficiency (N)"
                        )
                    else:
                        st.caption(f"📊 **Run Metrics:** Peaks ({latest_peaks}) == Spec ({peakspec}) | $t_R$ ({latest_runtime:.1f} min) <= Spec ({runspec} min) | $R_s$ ({latest_resolution:.2f}) >= Spec ({res_spec})")
                        st.success(
                            f"🎉 **Method Development Complete!** All specifications satisfied.\n\n"
                            f"*Peak Count:* {latest_peaks}/{peakspec} | *Run Time:* {latest_runtime:.1f} min (Target: ≤{runspec} min) | *Resolution:* {latest_resolution:.2f} (Target: ≥{res_spec})\n\n"
                            f"*Status:* Method MET. No further parameter adjustments required."
                        )

                # ------------------------------------------------------------------
                # CASE B: Peak count met, but Run Time or Resolution failed -> Direct to FC2
                # ------------------------------------------------------------------
                else:
                    if runtime_far_below and not pass_resolution:
                        st.caption(f"📊 **Run Metrics:** Peaks ({latest_peaks}) == Spec ({peakspec}) | $t_R$ ({latest_runtime:.1f} min) << Spec ({runspec} min) | $R_s$ ({latest_resolution:.2f}) < Spec ({res_spec})")
                        reason = (
                            f"Peak count met ({latest_peaks}/{peakspec}), but Resolution is below specification ({latest_resolution:.2f} < {res_spec}). "
                            f"Run time is very short ({latest_runtime:.1f} min < {0.5 * runspec:.1f} min)."
                        )
                    elif not pass_resolution and not pass_runtime:
                        st.caption(f"📊 **Run Metrics:** Peaks ({latest_peaks}) == Spec ({peakspec}) | $t_R$ ({latest_runtime:.1f} min) > Spec ({runspec} min) | $R_s$ ({latest_resolution:.2f}) < Spec ({res_spec})")
                        reason = (
                            f"Resolution is below specification ({latest_resolution:.2f} < {res_spec}) "
                            f"AND Run Time exceeds specification ({latest_runtime:.1f} min > {runspec} min)."
                        )
                    elif not pass_resolution:
                        st.caption(f"📊 **Run Metrics:** Peaks ({latest_peaks}) == Spec ({peakspec}) | $t_R$ ({latest_runtime:.1f} min) <= Spec ({runspec} min) | $R_s$ ({latest_resolution:.2f}) < Spec ({res_spec})")
                        reason = f"Resolution is below specification ({latest_resolution:.2f} < {res_spec})."
                    else:
                        st.caption(f"📊 **Run Metrics:** Peaks ({latest_peaks}) == Spec ({peakspec}) | $t_R$ ({latest_runtime:.1f} min) > Spec ({runspec} min) | $R_s$ ({latest_resolution:.2f}) >= Spec ({res_spec})")
                        reason = f"Run Time exceeds specification ({latest_runtime:.1f} min > {runspec} min)."

                    st.info(
                        f"**Next Run Strategy:** Direct to Efficiency Cycle (Flow Chart 2). %B and Selectivity parameters are frozen to preserve peak separation.\n\n"
                        f"*Reason:* {reason}\n\n"
                        f"*Controlled by:* Pump (Flow Rate) / Column (Dimensions, Particle Size) | *Category:* Mechanical | *Cycle:* Efficiency (N)"
                    )
        if st.session_state.get('show_decision_radio', False):
        choice1 = st.session_state.choice1
        choice2 = st.session_state.choice2
        # %B = (log10(k) - intercept) / slope
        # log(K) = (%B * slope) + intercept
        expected_intb_logk = choice2 * st.session_state.tr_slope + st.session_state.tr_intercept
        # K = (tR - t0)/t0
        # tR = K*t0 + t0
        expected_intb_tr = (10 ** expected_intb_logk) * st.session_state.tr_tzero + st.session_state.tr_tzero 
        choice_made = st.radio(
            f'Select whether to proceed with Intermediate %B ({choice2}%), or Minimum %B ({choice1}%):\n\n'
            f'Expected runtime at Intermediate %B is {expected_intb_tr} min. Expected runtime at Minimum %B: {runspec} min',
            ['Not Selected', 'Min%B', 'Intermediate %B']
        )
        print(choice1)
        print(choice2)
        if st.button('Confirm %B Selection'):
            selected_b = None
            if choice_made == 'Min%B':
                selected_b = choice1
            elif choice_made == 'Intermediate %B':
                selected_b = choice2
                
            if selected_b is not None:
                st.session_state.all_pct_bs.append(selected_b)
                
                static_keys = [
                        'all_solvents', 
                        'all_ligands', 
                        'all_bead_types',
                        'all_pore_sizes', 
                        'all_coreshells', 
                        'all_carbon_loads',
                        'all_column_lengths',      # Added missing key
                        'all_internal_diameters', 
                        'all_particle_sizes',
                        'all_flow_rates', 
                        'all_temperatures'
                ]

                # Propagate static chromatographic parameters to the next run
                for key in static_keys:
                    st.session_state[key].append(st.session_state[key][-1])
                    st.info(
                    f"**Next Run Strategy:** Change %B to {selected_b}%)\n\n"
                    f"*Controlled by:* Pump | *Parameter Type*: Chemical | *Cycle:* Retention"
                    )
                st.success(f"Added %B condition: {selected_b}% for next run!")
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
            use_container_width=True,
            hide_index=True
        )
    else:
        st.caption("No historical run records available yet. Upload a CSV chromatogram run to populate the record.")

    st.header('Equipment Interactive: Use embedded sheet below. Collapsed by default.')
    st.caption('URL to interactive, in case of embedding failure: https://docs.google.com/spreadsheets/d/1-Kl4UFp9Kk7IYOftbTtGVi-bior72YgTK0jYJZEbVK8/edit?usp=sharing')
    google_sheet_url = "https://docs.google.com/spreadsheets/d/1-Kl4UFp9Kk7IYOftbTtGVi-bior72YgTK0jYJZEbVK8/edit?gid=268052302#gid=268052302"

    with st.expander("📊 View Live Sheet Interactive", expanded=False):
        st.markdown(
            f"""
            <div style="
                position: relative;
                width: 100%;
                height: 600px;
                overflow: hidden;
                overscroll-behavior: contain;
                border: none;
                outline: none;
                margin: 0;
                padding: 0;
            ">
                <iframe 
                    src="{google_sheet_url}" 
                    loading="lazy"
                    tabindex="-1"
                    style="
                        position: absolute;
                        top: -120px;
                        left: 0;
                        width: 100%;
                        height: calc(100% + 120px);
                        border: none;
                        outline: none;
                    " 
                    frameborder="0"
                    scrolling="auto">
                </iframe>
            </div>
            """,
            unsafe_allow_html=True
        )

  if __name__ == "__main__":
    if not st.runtime.exists():
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())

    main()
