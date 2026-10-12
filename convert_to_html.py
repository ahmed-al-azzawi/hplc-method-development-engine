from script2stlite import Script2StliteConverter
import yaml

# Initialize the converter, pointing to your project directory
converter = Script2StliteConverter() # Or simply Script2StliteConverter() if running from this folder

# Convert the application
# You can optionally specify stlite_version and pyodide_version
# If not specified, the latest compatible versions will be used.
converter.convert()

# To print the exact output filename like above, you'd need to load settings first:
with open(f"settings.yaml", 'r') as f:
    settings = yaml.safe_load(f)

print(f"Conversion complete! Check for '{converter.directory}/{settings['APP_NAME'].replace(' ','_')}.html'.")
'''
This runs into issues when using streamlit.web, so the fix is to simply COMMENT on the #sys.exit(stcli.main()) and #from streamlit.web import cli as stcli
This is because normally streamlit runs with streamlit run app.py, but this way you can just run the script instead.
It converts toa  single HTML file, in case the deployed website is not running.
'''
