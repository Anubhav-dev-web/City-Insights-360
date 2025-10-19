state_pops_2023 = {
    'A & N Islands': 403000,
    'Andhra Pradesh': 53156000,
    'Andhra Pradesh Old': 53156000,  # Historical; we'll filter if needed
    'Arunachal Pradesh': 1562000,
    'Assam': 35713000,
    'Bihar': 126756000,
    'Chandigarh': 1231000,
    'Chhattisgarh': 30180000,
    'Dadra & Nagar Haveli': 1263000,
    'Daman & Diu': 1263000,  # Merged UT
    'Delhi': 21359000,
    'Goa': 1575000,
    'Gujarat': 71507000,
    'Haryana': 30209000,
    'Himachal Pradesh': 7468000,
    'Jammu & Kashmir': 13603000,
    'Jharkhand': 39466000,
    'Karnataka': 67692000,
    'Kerala': 35776000,
    'Lakshadweep': 69000,
    'Madhya Pradesh': 86579000,
    'Maharashtra': 126385000,
    'Manipur': 3223000,
    'Meghalaya': 3349000,
    'Mizoram': 1238000,
    'Nagaland': 2233000,
    'Odisha': 46276000,
    'Puducherry': 1646000,
    'Punjab': 30730000,
    'Rajasthan': 81025000,
    'Sikkim': 689000,
    'Tamil Nadu': 76860000,
    'Telangana': 38090000,
    'Tripura': 4147000,
    'Uttar Pradesh': 235687000,
    'Uttarakhand': 11637000,
    'West Bengal': 99084000
}
import pandas as pd
from pathlib import Path
import sys


def find_healthcare_csv():
    """Locate the healthcare CSV in a few likely places and return a Path.

    Search order:
    - same folder as this script
    - repo_root/powerbi_dashboard/healthcare_metrics.csv
    - any file in repo matching '*healthcare*.csv'
    """
    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parents[1] if len(this_dir.parents) >= 2 else this_dir

    candidates = []
    candidates.append(this_dir / 'healthcare_metrics.csv')
    candidates.append(repo_root / 'powerbi_dashboard' / 'healthcare_metrics.csv')

    # Add any other matching files under the repo root
    try:
        for p in repo_root.rglob('*healthcare*.csv'):
            candidates.append(p)
    except Exception:
        # In some restricted environments rglob may fail; ignore those errors
        pass

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for p in candidates:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if rp not in seen:
            seen.add(rp)
            uniq.append(p)

    for p in uniq:
        if p.exists():
            return p

    # If none found, show helpful diagnostics
    tried = '\n'.join(str(p) for p in uniq)
    raise FileNotFoundError(
        f"healthcare CSV not found. Tried these locations:\n{tried}\n\n"
        "Place 'healthcare_metrics.csv' in the same folder as this script or in 'powerbi_dashboard/'"
    )


# Load healthcare data (will search for the file)
try:
    healthcare_csv = find_healthcare_csv()
    print(f"Using healthcare CSV: {healthcare_csv}")
    healthcare_df = pd.read_csv(healthcare_csv)
except FileNotFoundError as e:
    print(str(e), file=sys.stderr)
    sys.exit(2)

# Normalize column names (strip whitespace)
healthcare_df.columns = [c.strip() for c in healthcare_df.columns]

# Determine state column (common variants)
state_col = None
for candidate in ('State Name', 'State_Name', 'State/UT', 'State/UT/Division', 'State_Name'):
    if candidate in healthcare_df.columns:
        state_col = candidate
        break
if state_col is None:
    raise KeyError('Could not find a State column in the healthcare CSV. Columns found: ' + ','.join(healthcare_df.columns))

# Compute total facilities per state if not provided
if 'total_facilities' in healthcare_df.columns:
    # assume column already provides totals per-row (maybe state-level)
    state_totals = healthcare_df.groupby(state_col)['total_facilities'].sum()
else:
    # Prefer counting unique facility names when available, otherwise count rows
    if 'Facility Name' in healthcare_df.columns:
        state_totals = healthcare_df.groupby(state_col)['Facility Name'].nunique()
    else:
        state_totals = healthcare_df.groupby(state_col).size()

state_df = state_totals.reset_index().rename(columns={0: 'total_facilities', state_col: 'State'})
if 'total_facilities' not in state_df.columns:
    state_df = state_df.rename(columns={state_df.columns[1]: 'total_facilities'})

# Map population
state_df['Population'] = state_df['State'].map(state_pops_2023)

# Drop states without population data
state_df = state_df.dropna(subset=['Population'])

# Calculate facilities per 1 lakh population
state_df['Facilities per 1L Pop'] = state_df['total_facilities'] / (state_df['Population'] / 100000)

# Optional: Filter out 'Old' entries for cleanliness
state_df = state_df[~state_df['State'].str.contains('Old', na=False)]

# Quick stats (state-level)
avg_density = state_df['Facilities per 1L Pop'].mean()
print(f"Average (state-level): {avg_density:.2f} per 1L")

# Export enriched CSV
export_dir = Path(__file__).resolve().parent / 'powerbi_dashboard'
export_dir.mkdir(parents=True, exist_ok=True)
state_df.to_csv(export_dir / 'healthcare_enriched.csv', index=False)
print(f"Exported to {export_dir / 'healthcare_enriched.csv'}")