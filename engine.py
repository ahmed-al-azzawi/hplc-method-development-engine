

# Necessary imports
import sys
from streamlit.web import cli as stcli
import streamlit as st
import pandas as pd
import chardet
import numpy as np
import io


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
  
  if __name__ == "__main__":
    if not st.runtime.exists():
        sys.argv = ["streamlit", "run", __file__]
        sys.exit(stcli.main())

    main()
