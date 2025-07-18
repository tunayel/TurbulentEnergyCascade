# main.py

import os
from analyze_case import analyze_case  # this is your main analysis function

# === 1. List all input files ===
# Folder where CSV files are stored
data_folder = "cases"

# File list: (filename, label) tuples
cases = [
    ("2500_mesh.csv", "2500 RPM with mesh"),
    ("2500_nomesh.csv", "2500 RPM without mesh"),
    ("5000_mesh.csv", "5000 RPM with mesh"),
    ("5000_nomesh.csv", "5000 RPM without mesh"),
    ("7500_mesh.csv", "7500 RPM with mesh"),
    ("7500_nomesh.csv", "7500 RPM without mesh")
]

# === 2. Output Folder ===
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

# === 3. Loop through each case ===
for filename, label in cases:
    filepath = os.path.join(data_folder, filename)
    print(f"\n🚀 Running analysis for: {label}")
    analyze_case(filepath, label, output_folder)
