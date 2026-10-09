# How to run this

Shortest path to reproducing what's in the repo. Written as the notes I'd want when I come back to this after a break.

## What you need

- Python 3.9 or newer
- About 8 GB of RAM to run `01_data_cleaning.ipynb` comfortably — it loads the ~1.5 GB raw CSV before sampling it down
- The Lending Club dataset from Kaggle: <https://www.kaggle.com/datasets/wordsforthewise/lending-club>

I use `venv` on macOS. `conda` should work; I just haven't tested that path.

## Setup

```bash
git clone https://github.com/Evelynval-w/credit_risk_ML_model.git
cd credit_risk_ML_model
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Download `accepted_2007_to_2018Q4.csv` from the Kaggle link above and drop it in `data/raw/`. That folder is empty in the repo on purpose — the file is too big for GitHub (my `.gitignore` keeps `data/raw/*.csv` out).

## Running the notebooks

Two exist right now. They have to run in order because 02 reads what 01 writes.

```bash
jupyter notebook
```

1. **`notebooks/01_data_cleaning.ipynb`** — loads the raw CSV, keeps 32 columns, builds the binary target, samples down to 250k rows stratified by target, converts string columns to the right types, documents the missing-value pattern (but deliberately doesn't impute — that lives in 03, after the train/test split), writes `data/processed/loan_cleaned.csv`.
2. **`notebooks/02_eda.ipynb`** — reads the cleaned CSV, generates the charts in `images/` and writes three per-segment summary CSVs under `data/processed/` for later use in a dashboard tool.

Notebooks 03–05 (feature engineering, modeling, business metrics) don't exist yet. Those are what I'm building next.

## What gets created

```
data/processed/
├── loan_cleaned.csv              # from 01
├── summary_by_grade.csv          # from 02
├── summary_by_purpose.csv        # from 02
└── summary_by_state.csv          # from 02

images/
├── 01_target_distribution.png
├── 02_numerical_distributions.png
├── 03_default_by_grade.png
├── 03_default_by_grade_interactive.html
├── 04_correlation_with_target.png
├── 05_correlation_heatmap.png
├── 06_default_by_purpose.png
├── 07_default_by_home_ownership.png
├── 08_interest_rate_analysis.png
├── 09_default_by_fico.png
├── 10_default_by_state_map.html
├── 11_loan_amount_analysis.png
└── 12_eda_dashboard.html
```

## Things that have actually bitten me

- **`earliest_cr_line` date parsing.** Lending Club writes dates like `Aug-2003` (four-digit year). If you pass `format='%b-%y'` pandas silently misparses them. The notebook uses `format='%b-%Y'`.
- **`low_memory=True` on the raw CSV.** The pandas default chokes on the mixed-type columns in this file and raises confusing dtype warnings. Explicit `low_memory=False` on `pd.read_csv` is what works.
- **Plotly figures not rendering in classic Jupyter.** They render in JupyterLab, and the `.html` files are saved regardless, so it's not blocking — just know you may need to open the HTMLs in a browser if the inline cell output is blank.
