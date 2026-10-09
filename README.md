# Credit risk model — Lending Club

I'm Okoene Makuochukwu, a CS student at EPITA. I'm working through Andrew Ng's Machine Learning Specialization (Coursera, Stanford + DeepLearning.AI) and this repo is where I take each week's ideas and try them on a real dataset instead of the course's toy data. Right now it's a credit-risk project on Lending Club loans.

## Where this is right now (October 2026)

Cleaning and EDA are done. The actual modeling isn't — I'm partway through the project and this README reflects what's shipped, not what's planned.

- **Done:** `01_data_cleaning.ipynb`, `02_eda.ipynb`.
- **Not yet:** `03_feature_engineering.ipynb`, `04_modeling.ipynb`, `05_business_metrics.ipynb`. These are what I'm working on next.
- **No metrics in this README yet** — I'll put real ROC-AUC / PR-AUC / expected-loss numbers here once 04 lands. I'd rather have no numbers than invented ones.

## The question

Can we predict, at the time a loan is issued, whether a Lending Club borrower will default? It's a binary classification problem, and the thing that makes it interesting is that defaults are rare and the cost of a false negative (approving someone who then defaults) is much larger than the cost of a false positive (rejecting someone who would have paid). Andrew's lecture on skewed classes and the precision-recall tradeoff is more or less the shape of this problem.

## Dataset

Lending Club public loan data, 2007–2018, ~2.26 million rows and 145 columns. I kept 32 of those columns — the ones I thought could plausibly be known at the time of issuance — removed identifiers and post-loan columns like `total_pymnt` and `recoveries` (which would leak the outcome into the features), filtered to loans whose outcome is known (so, dropped "Current"), and took a stratified sample of 250k rows to keep the notebooks responsive on a laptop.

## What I did so far

### 01 — cleaning

Three decisions here were the ones I actually had to think about:

1. **How to define "default".** Lending Club's `loan_status` has about ten values. I grouped `Charged Off`, `Default`, `Late (31–120 days)`, `Late (16–30 days)` and the two `Does not meet the credit policy...Charged Off` variants as the positive class. Lumping the "late" statuses in with the actually-charged-off loans is a judgment call — those borrowers aren't technically defaults yet. I chose to include them because for a credit team the operational decision point is "is this loan showing distress", not "has it already failed". I'll revisit this when I look at calibration in 04.
2. **How to sample.** The full set is manageable but slow to iterate on, so I stratified-sampled 250k rows by target. This preserves the class ratio but shrinks the long tail of rare states and purposes — something to keep in mind when the geographic chart in 02 shows noisy tiles for the smaller states.
3. **What to drop as leakage.** Anything that could only exist after the loan matured — `total_pymnt`, `recoveries`, `last_pymnt_*` — came out. Andrew makes a big deal about leakage in the data-preparation lecture; this was the first time I actually had to apply that discipline on real data rather than a cleaned toy dataset.

Two things I need to fix before modeling, which I caught while reviewing my own notebook:

- The median imputation in cell 21 is fit on the whole dataset. I need to do the train/test split first and fit the imputer on train only. This is exactly the leakage pattern Andrew warns about.
- `credit_history_years` currently uses a hard-coded `2018-12-31` as the "as-of" date instead of each loan's own `issue_d`. For older loans that gives the model future information. The fix is to compute `(issue_d − earliest_cr_line)` row by row.

Both are easy fixes, but I want to call them out rather than quietly patch them — the point of this project is to catch my own mistakes, not pretend I didn't make any.

### 02 — EDA

The EDA notebook is organized around one question: of the features we kept, which actually separate defaulters from non-defaulters, and by how much?

A few things stood out:

- Grade is the single clearest signal. Default rate climbs almost monotonically from grade A (around 5%) to grade G (around 30%). Lending Club's grade already encodes a lot of what we'd want to learn, so part of what 04 will have to tell us is how much *additional* signal is left in the raw features once grade is in.
- Interest rate tracks default roughly linearly. That's risk-based pricing working as advertised. For modeling, it means `int_rate` and `grade` carry overlapping information — something to think about when I regularize the logistic regression.
- FICO is the obvious inverse. Default rate drops from about 22% at FICO <650 to about 5% at 800+. No surprise, but worth confirming on this specific sample before relying on it.

One thing to flag: I did EDA on the full sampled dataset, not just a training portion. The clean move is to hold out the test set *before* looking at it, so EDA observations can't bias feature choices. I'm going to redo these grouped statistics on only the train split in 03.

## What's coming next

- **03 — feature engineering.** Time-aware split by `issue_d` (train on 2007–2017, hold out 2018). Ratios (loan-to-income, debt burden, FICO midpoint), encodings for the categorical features, standardization only for the LR model. These features were originally in 01; I'm moving them so the split happens before any feature construction that pools statistics across rows.
- **04 — modeling.** Baseline logistic regression first, then regularized LR, then random forest, then XGBoost. Learning curves for each so I can actually diagnose high-bias vs high-variance the way Andrew teaches it. Class imbalance handled with `class_weight='balanced'` first, and SMOTE as a comparison.
- **05 — business metrics.** Expected loss (PD × LGD × EAD), threshold tuning as a cost-sensitive decision, and a short note on what the model would actually tell a credit team.

## How I'm learning this

I'm working through the second course of the specialization (Advanced Learning Algorithms). The parts of the course that most directly shape what you'll see in this repo:

- **Bias/variance diagnosis via learning curves** — how I'll compare the four models in 04.
- **Cost functions and skewed classes** — why 04 will report PR-AUC alongside ROC-AUC, and why 05 picks the decision threshold on expected cost rather than on the ROC curve's elbow.
- **Regularization** — why I'll start with L2-regularized logistic regression before touching the ensembles.
- **Data leakage** — the two bugs I called out above, and the time-aware split in 03.

## Running it yourself

See [`HOW_TO_RUN.md`](./HOW_TO_RUN.md).

## A few of the EDA charts

Default rate by grade:

![Default rate by grade](./images/03_default_by_grade.png)

Feature correlation with the target — red raises default risk, green lowers it:

![Correlation with target](./images/04_correlation_with_target.png)

Interest rate against loan outcome:

![Interest rate analysis](./images/08_interest_rate_analysis.png)

The interactive versions (grade bar, choropleth, four-panel dashboard) are in `images/` as `.html` files.

## Contact

- GitHub: [@Evelynval-w](https://github.com/Evelynval-w)
- Email: okoeneeve@gmail.com
- LinkedIn: [okoene-makuo-56932b259](https://www.linkedin.com/in/okoene-makuo-56932b259/)
