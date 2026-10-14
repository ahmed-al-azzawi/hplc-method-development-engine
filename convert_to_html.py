from script2stlite import Script2StliteConverter
import yaml

# 1. Load settings explicitly
with open("settings.yaml", "r") as f:
    settings = yaml.safe_load(f)

# 2. Initialize and convert
converter = Script2StliteConverter()
converter.convert()

output_filename = settings["APP_NAME"].replace(" ", "_") + ".html"
print(f"\n✅ Conversion complete! Output file: '{output_filename}'")
