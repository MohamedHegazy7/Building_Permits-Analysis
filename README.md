<!-- README for San Francisco Building Permits Project -->
# San Francisco Building Permits — Data Exploration

## Project Overview
- **Purpose:** Explore and document the San Francisco building permits dataset. The workspace contains an analysis notebook, Streamlit app pages for univariate and bivariate exploration, and a data-cleaning script.

## Dataset Description
- **Source file:** `Building_Permits.csv` (original dataset used across the notebook and Streamlit pages).
- **Rows / Columns:** The dataset contains building-permit level records with ~43 columns covering identification, status, timeline dates, location, building details (existing vs proposed), financials, and sparse flags (e.g., Structural Notification, Fire Only Permit).
- **Key column groups:**
  - Identification & Status: Permit Number, Record ID, Permit Type, Current Status
  - Timeline: Permit Creation Date, Filed Date, Issued Date, Completed Date, First Construction Document Date, Permit Expiration Date
  - Location: Block, Lot, Street Number, Street Name, Street Suffix, Zipcode, Location, Supervisor District, Neighborhoods - Analysis Boundaries
  - Building Details: Number of Existing Stories, Number of Proposed Stories, Existing/Proposed Use, Existing/Proposed Units, Construction Type and Descriptions
  - Financials & Plans: Estimated Cost, Revised Cost, Plansets
  - Flags: Structural Notification, Voluntary Soft-Story Retrofit, Fire Only Permit, TIDF Compliance, Site Permit

## What the Notebook Contains
- **File:** `Data_Understanding_and_Exploration.ipynb` — step-by-step EDA and cleaning. Major notebook sections:
  - Data Understanding: column listing, data dictionary, initial head and info checks.
  - Data Exploration: summary statistics, missing-value analysis, duplicates, and visual checks (histograms, bar charts, pie charts).
  - Data Cleaning: column drops, imputations (median/mode/KNN for numerics), filling categorical missing values, date parsing (`pd.to_datetime`) and conversion for date columns.
  - Outlier handling: IQR filtering for selected numeric columns.
  - Feature engineering: Processing time calculation (Issued Date - Filed Date) and derived analyses (neighborhood + permit type heatmaps, cost comparisons).
  - Save step: a cleaned CSV (`cleaned_df.csv`) is produced for downstream usage.

## Streamlit App Files
- **Top-level:** `Home.py` — main dashboard with dataset overview, column descriptions, previews, and summary statistics.
- **Pages folder:**
  - `pages/Univariate.py` — univariate exploration (numeric: histogram & box; categorical: bar & pie, top-K grouping).
  - `pages/Bivariate.py` — bivariate analysis (scatter, box, density heatmap, aggregated bars, categorical heatmaps), with sampling and top-K truncation for performance.

## Data Cleaning Notes
- Date columns (e.g., `Filed Date`, `Issued Date`, `Permit Creation Date`, etc.) are converted from object to datetime where possible using `pd.to_datetime(errors='coerce')`.
- Columns with high missingness were considered for dropping; medium missingness for imputation (median for discrete numeric, KNN imputer for continuous numeric), and categorical fills used `'Unknown'` for sparse fields.
- Mixed-type columns may trigger pandas DtypeWarning; the notebook documents detection and resolution strategies.

## Deployment & How to Run
1. Create and activate a Python environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # PowerShell
```

2. Install dependencies (example `requirements.txt`):

```powershell
pip install streamlit pandas plotly scikit-learn
```

3. Run the main Streamlit app from the project root:

```powershell
streamlit run Home.py
```

4. To run a specific page (alternative):

```powershell
streamlit run pages/Univariate.py
streamlit run pages/Bivariate.py
```

## Notes & Next Steps
- If the dataset is large, Streamlit pages include sampling and top-K truncation to keep charts responsive; adjust sampling parameters in the page code if you need full-detail plots.
- Optionally add a `requirements.txt` with pinned versions and/or a `README` section for export/download instructions (CSV export from filtered views) if required.

## Files of Interest
- `Building_Permits.csv` — raw dataset
- `cleaned_df.csv` — cleaned dataset saved by notebook
- `Data_Understanding_and_Exploration.ipynb` — EDA and cleaning notebook
- `Home.py`, `pages/Univariate.py`, `pages/Bivariate.py` — Streamlit app and pages

---
Prepared from the project notebook and Streamlit scripts to help run and understand the analysis and deployment.
