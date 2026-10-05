# feature-engineering

Notes and worked solutions for **[Feature Engineering](https://www.kaggle.com/learn/feature-engineering)** (Ryan Holbrook) — the fourth course here, after `pandas-learn/`, `datacleaning-learn/`, and `intro-machine-learning/`.

*Builds on: Pandas, Intro to Machine Learning. Preparation for: the House Prices competition, and the course's own later sections on clustering, PCA, and target encoding.*

## Sections

| # | Tutorial | Exercise | Questions |
| --- | --- | --- | --- |
| 1 | [What Is Feature Engineering](https://www.kaggle.com/code/ryanholbrook/what-is-feature-engineering) | [exercise](https://www.kaggle.com/code/ryanholbrook/exercise-what-is-feature-engineering) — *not done* | — |
| 2 | [Mutual Information](https://www.kaggle.com/code/ryanholbrook/mutual-information) | [exercise](https://www.kaggle.com/code/ryanholbrook/exercise-mutual-information) | 3 |

Sections 3–6 (Creating Features, Clustering With K-Means, Principal Component Analysis, Target Encoding) plus the bonus House Prices notebook are not started — all of them live on [the course page](https://www.kaggle.com/learn/feature-engineering).

Naming follows the other directories: `NN-<topic>.ipynb` is the lesson walkthrough, `NN-exercise.ipynb` is the exercise with answers filled in. One difference from the earlier courses: the lesson notebooks here are **code only** — no markdown cells, the narrative stays on the Kaggle tutorial page, so read them side by side with it.

| File | What it is |
| --- | --- |
| `01-what-is-feature-engineering.ipynb` | Lesson walkthrough — load the Concrete dataset, baseline `RandomForestRegressor` with `cross_val_score`, then add `FCRatio` / `AggCmtRatio` / `WtrCmtRatio` synthetic features and re-score |
| `02-mutual-information.ipynb` | Lesson walkthrough — `Exter Qual` vs `SalePrice` strip plot on Ames, then the full MI machinery on the Automobile dataset: word→number mapping with `.map()`, `factorize()` label encoding, `discrete_features` dtype check, `mutual_info_regression`, `plot_mi_scores`, `relplot`/`lmplot` interaction checks |
| `02-exercise.ipynb` | Exercise 2 solutions, questions 1–3 (which of `YearBuilt`/`MoSold`/`ScreenPorch` has highest MI, reading the MI score themes, `BldgType` interaction with `GrLivArea` vs `MoSold`) |

## Datasets

All from Kaggle. Inputs live in `datasets/{dataset-slug}/` and are referenced from the notebooks as `../datasets/…`:

| Used by | Dataset | Path | Files |
| --- | --- | --- | --- |
| Lesson 1 | [Concrete Compressive Strength](https://www.kaggle.com/datasets/sinamhd9/concrete-comprehensive-strength) | `datasets/concrete-comprehensive-strength/` | `concrete_data.csv` (1,029 rows, 58 KB) |
| Lesson 2 (first plot) | [Ames Housing](https://www.kaggle.com/datasets/ameroland/ames-housing) — the original De Cock file | `datasets/ames-housing-dataset/` | `AmesHousing.csv` (2,930 rows, 82 cols, 941 KB) |
| Exercise 2 | *Ames*, same rows with `Order`/`PID` dropped and columns renamed to the `data_description.txt` style — the course's own copy, from the [House Prices competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) | `datasets/ames-housing-dataset/` | `ames.csv` (2,930 rows, 79 cols = 78 features + `SalePrice`, 1.4 MB) |
| Lesson 2 (MI demo) | [Automobile Dataset](https://www.kaggle.com/datasets/toramky/automobile-dataset) | `datasets/automobile-dataset/` | `Automobile_data.csv` (205 rows, 24 KB) |

Every input CSV under `datasets/` is gitignored — datasets are downloaded from Kaggle, never committed. These notebooks write no CSVs of their own, so `feature-engineering/` needs no extra `.gitignore` entry.

Two gotchas the notebooks work around: the concrete file's column is spelled `fine_aggregate ` (trailing space — quote it exactly), and `Automobile_data.csv` stores `?` for missing values plus `num-of-doors`/`num-of-cylinders` as words (`"four"`), so it gets `.map()`-ed to numbers and `pd.to_numeric(errors="coerce")` before anything else.

## Running the notebooks

```bash
# from the repo root
.venv/bin/jupyter lab feature-engineering/
```

Notebooks read their CSVs from `../datasets/`, one level up from the notebook.

### Dependencies

| Package | Used by |
| --- | --- |
| `pandas` | every notebook |
| `numpy` | lesson 2 (`np.nan` in the word→number map), exercise (`np.arange` for bar positions) |
| `matplotlib`, `seaborn` | lesson 2 & exercise — `sns.set_theme`, `catplot`/`relplot`/`lmplot`, `plot_mi_scores` |
| `scikit-learn` | lesson 1 (`RandomForestRegressor`, `cross_val_score`), lesson 2 + exercise (`mutual_info_regression`) |

Installed in the repo venv: Python 3.14.8, pandas 3.0.6, numpy 2.5.3, matplotlib 3.11.2, seaborn 0.13.2, scikit-learn 1.9.1.

## Concepts covered

### 1. What Is Feature Engineering

- **The goal** is to make the data better suited to the problem — improve predictive performance, reduce data needs, or make results easier to interpret
- **Guiding principle**: a feature is useful only if the model can learn its relationship to the target. A transformation becomes part of the model itself — squaring `Length` into `Area` is what lets a *linear* model fit a parabola
- **Baseline first** — score the untouched dataset before inventing anything, so you can tell whether new features earned their place: `cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")`, negated because scikit-learn's default convention is "higher is better"
- **Synthetic features** from domain reasoning: concrete strength depends on ingredient *ratios*, not absolute amounts, so add `FCRatio` (fine/coarse aggregate), `AggCmtRatio` (aggregate/cement), `WtrCmtRatio` (water/cement)
- `df.copy()` + `X.pop("concrete_compressive_strength")` is the split used throughout — keep `X` and `y` in step when you add columns

### 2. Mutual Information

- **Mutual information measures dependence, not correlation** — it detects *any* relationship, linear or not, and is 0 only when the features are independent
- **Pipeline**: label-encode categoricals with `X[colname], _ = X[colname].factorize()`, declare `discrete_features = X.dtypes == int` (or a per-column dtype list), then `mutual_info_regression(X, y, discrete_features=...)`
- The dtype check matters: MI treats a column as continuous or discrete based on that flag, and a label-encoded categorical left as `object` will fail or be mishandled
- Scores are non-negative and unbounded; sort descending and plot with `plt.barh` (`plot_mi_scores`) — the *ranking* is what you read, not the raw number
- **Interaction effects**: a low overall MI score can hide a useful feature. `BldgType` barely separates `SalePrice` on its own, but its trend lines against `GrLivArea` differ sharply per category — knowing `BldgType` changes how `GrLivArea` relates to the target, so keep it. The same test against `MoSold` shows nearly identical lines → no interaction
- MI is a filter for *where to look*, not a model: scores are noisy estimates, are not comparable across different targets, and high-cardinality categoricals can score well just by being many-valued

## Results

Lesson 1, 5-fold CV MAE on Concrete:

| Stage | MAE |
| --- | --- |
| Baseline `RandomForestRegressor` | 8.38 |
| + `FCRatio`, `AggCmtRatio`, `WtrCmtRatio` | 7.949 |

Exercise 2, top 10 MI scores for `SalePrice` over the 78 Ames features:

| Feature | MI | Theme |
| --- | --- | --- |
| `OverallQual` | 0.581 | quality |
| `Neighborhood` | 0.570 | location |
| `GrLivArea` | 0.497 | size |
| `YearBuilt` | 0.438 | year |
| `GarageArea` | 0.415 | size |
| `TotalBsmtSF` | 0.390 | size |
| `GarageCars` | 0.381 | size |
| `FirstFlrSF` | 0.369 | size |
| `BsmtQual` | 0.365 | quality |
| `KitchenQual` | 0.326 | quality |

The bottom of the ranking — `MoSold`, `LandSlope`, `Threeseasonporch`, `BsmtFinSF2` all at 0.000 — is mostly rare or exceptional conditions that don't describe an average home. Lesson 2's Automobile run ranks `curb-weight` (1.468) and `horsepower` (0.847) far above everything else, with `num-of-doors` at 0.000.

## Next

Course: [kaggle.com/learn/feature-engineering](https://www.kaggle.com/learn/feature-engineering) — discussion forum [here](https://www.kaggle.com/learn/feature-engineering/discussion).

Four courses now done: `pandas-learn/` (6 sections), `datacleaning-learn/` (5 sections), `intro-machine-learning/` (7 sections), and this one (2 of 6 sections). Next here is **Creating Features** — the lesson this exercise explicitly hands off to — then Clustering With K-Means, PCA, and Target Encoding, which close out with a submission to the House Prices competition.
