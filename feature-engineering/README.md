# feature-engineering

Notes and worked solutions for **[Feature Engineering](https://www.kaggle.com/learn/feature-engineering)** (Ryan Holbrook) — the fourth course here, after `pandas-learn/`, `datacleaning-learn/`, and `intro-machine-learning/`.

*Builds on: Pandas, Intro to Machine Learning. Preparation for: the House Prices competition, and the course's later sections on PCA and target encoding.*

## Sections

| # | Tutorial | Exercise | Questions |
| --- | --- | --- | --- |
| 1 | [What Is Feature Engineering](https://www.kaggle.com/code/ryanholbrook/what-is-feature-engineering) | [exercise](https://www.kaggle.com/code/ryanholbrook/exercise-what-is-feature-engineering) — *not done* | — |
| 2 | [Mutual Information](https://www.kaggle.com/code/ryanholbrook/mutual-information) | [exercise](https://www.kaggle.com/code/ryanholbrook/exercise-mutual-information) | 3 |
| 3 | [Creating Features](https://www.kaggle.com/code/ryanholbrook/creating-features) | [exercise](https://www.kaggle.com/code/ryanholbrook/exercise-creating-features) | 5 |
| 4 | [Clustering With K-Means](https://www.kaggle.com/code/ryanholbrook/clustering-with-k-means) | [exercise](https://www.kaggle.com/code/ryanholbrook/exercise-clustering-with-k-means) | 3 |

Sections 5–6 (Principal Component Analysis, Target Encoding) plus the bonus House Prices notebook are not started — all of them live on [the course page](https://www.kaggle.com/learn/feature-engineering).

Naming follows the other directories: `NN-<topic>.ipynb` is the lesson walkthrough, `NN-exercise.ipynb` is the exercise with answers filled in. One difference from the earlier courses: the lesson notebooks here are **code only** — no markdown cells, the narrative stays on the Kaggle tutorial page, so read them side by side with it.

| File | What it is |
| --- | --- |
| `01-what-is-feature-engineering.ipynb` | Lesson walkthrough — load the Concrete dataset, baseline `RandomForestRegressor` with `cross_val_score`, then add `FCRatio` / `AggCmtRatio` / `WtrCmtRatio` synthetic features and re-score |
| `02-mutual-information.ipynb` | Lesson walkthrough — `Exter Qual` vs `SalePrice` strip plot on Ames, then the full MI machinery on the Automobile dataset (`autos.csv`): `factorize()` label encoding, `discrete_features` dtype check, `mutual_info_regression`, `plot_mi_scores`, `relplot`/`lmplot` interaction checks |
| `02-exercise.ipynb` | Exercise 2 solutions, questions 1–3 (which of `YearBuilt`/`MoSold`/`ScreenPorch` has highest MI, reading the MI score themes, `BldgType` interaction with `GrLivArea` vs `MoSold`) |
| `03-creating-features.ipynb` | Lesson walkthrough — ratio/displacement features on autos, `np.log1p` on skewed `WindSpeed`, boolean-count features (roadway flags, concrete components), `str.split` to break apart `Policy`, `make_body_style` concatenation, `groupby().transform()` for `AverageIncome` / `StateFreq`, and a train-only grouped mean merged onto validation |
| `03-exercise.ipynb` | Exercise 3 solutions, questions 1–5 (`LivLotRatio`/`Spaciousness`/`TotalOutsideSF`, `BldgType`×`GrLivArea` one-hot interaction, `PorchTypes` count, `MSClass` split, `MedNhbdArea` grouped median) + before/after `score_dataset` |
| `04-clustering-with-k-means.ipynb` | Lesson walkthrough — `KMeans(n_clusters=6)` fit on `MedInc`/`Latitude`/`Longitude` from California housing, `fit_predict` cast to `category` as a `Cluster` feature, `relplot` map of cluster membership, `catplot(kind="boxen")` of `MedHouseVal` per cluster |
| `04-exercise.ipynb` | Exercise 4 solutions, questions 1–3 (when to rescale before clustering — three feature sets reasoned through, 10-cluster labels from five *scaled* Ames area features with `n_init=10, random_state=0`, then the ten `fit_transform` centroid-distance features) + `score_dataset` after each |

## Datasets

All from Kaggle. Inputs live in `datasets/{dataset-slug}/` and are referenced from the notebooks as `../datasets/…`:

| Used by | Dataset | Path | Files |
| --- | --- | --- | --- |
| Lesson 1 | [Concrete Compressive Strength](https://www.kaggle.com/datasets/sinamhd9/concrete-comprehensive-strength) — the raw UCI copy | `datasets/concrete-comprehensive-strength/` | `concrete_data.csv` (1,029 rows, 58 KB) |
| Lesson 3 | same dataset, the course's cleaned copy (`Cement`, `FineAggregate`, …) | `datasets/concrete-comprehensive-strength/` | `concrete.csv` (1,030 rows, 9 cols, 48 KB) |
| Lessons 2–3 | [Ames Housing](https://www.kaggle.com/datasets/ameroland/ames-housing) — the original De Cock file | `datasets/ames-housing-dataset/` | `AmesHousing.csv` (2,930 rows, 82 cols, 941 KB) |
| Exercises 2–3 | *Ames*, same rows with `Order`/`PID` dropped and columns renamed to the `data_description.txt` style — the course's own copy, from the [House Prices competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) | `datasets/ames-housing-dataset/` | `ames.csv` (2,930 rows, 79 cols = 78 features + `SalePrice`, 1.4 MB) |
| Lessons 2–3 | [Automobile Dataset](https://www.kaggle.com/datasets/toramky/automobile-dataset) — the course's cleaned copy (`curb_weight`, no `?`) | `datasets/automobile-dataset/` | `autos.csv` (193 rows, 25 cols, 24 KB); `Automobile_data.csv` (205 rows, 24 KB) is the raw UCI original, left over from the first run of lesson 2 and no longer read |
| Lesson 3 | [US Accidents (2016–2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) — the 100k-row course sample | `datasets/us-accidents/` | `accidents.csv` (100,000 rows, 29 cols, 20 MB) |
| Lesson 3 | [IBM Watson Marketing Customer Value Data](https://www.kaggle.com/datasets/pankajjsh06/ibm-watson-marketing-customer-value-data) | `datasets/ibm-watson-marketing-customer-value-data/` | `customer.csv` (9,134 rows, 25 cols, 1.6 MB) |
| Lesson 4 | [California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices) | `datasets/california-housing-prices/` | `housing.csv` (20,640 rows, 9 cols, 1.9 MB) |

Every input CSV under `datasets/` is gitignored — datasets are downloaded from Kaggle, never committed. These notebooks write no CSVs of their own, so `feature-engineering/` needs no extra `.gitignore` entry.

Gotchas the notebooks work around: lesson 1 reads the raw concrete file, whose column is spelled `fine_aggregate ` (trailing space — quote it exactly), while lesson 3 reads the cleaned `concrete.csv` where the same column is `FineAggregate`. Same story for cars — `Automobile_data.csv` stores `?` for missing values and `num-of-doors` as words (`"four"`), which is why the original run of lesson 2 needed the `.map()` / `pd.to_numeric(errors="coerce")` block; that block is now commented out because the notebooks switched to the course's clean `autos.csv`.

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
| `numpy` | lessons 2 & 3 (`np.log1p`, `np.pi` for displacement), exercises (`np.sqrt` in `score_dataset`, `np.arange` for bar positions) |
| `matplotlib`, `seaborn` | lessons 2–4 & exercises — `sns.set_theme`, `catplot`/`relplot`/`lmplot`, `kdeplot`, `boxen`, `plot_mi_scores` |
| `scikit-learn` | lessons 1–2 & 4, exercises 2–4 (`RandomForestRegressor`, `cross_val_score`, `mutual_info_regression`, `KMeans`) |
| `xgboost` | exercises 3–4 — `XGBRegressor` inside `score_dataset` |

Installed in the repo venv: Python 3.14.8, pandas 3.0.6, numpy 2.5.3, matplotlib 3.11.2, seaborn 0.13.2, scikit-learn 1.9.1, xgboost 3.4.1.

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

### 3. Creating Features

- **Ratios and physics**: `stroke_ratio = stroke / bore` and `displacement = π × (bore/2)² × stroke × num_of_cylinders` — combine existing columns into a quantity the domain actually cares about instead of leaving the model to find it
- **Reshape a skewed distribution**: `accidents["LogWindSpeed"] = accidents.WindSpeed.apply(np.log1p)`; compare the two `kdeplot`s — the long right tail collapses toward symmetric, which is what most models want
- **Count features**: `df[roadway_flags].sum(axis=1)` counts how many of 12 booleans are true (`RoadwayFeatures`), `concrete[components].gt(0).sum(axis=1)` counts how many ingredients are present (`Components`) — `.sum()` works on booleans because they're 1/0
- **Break a categorical apart**: `customer["Policy"].str.split(" ", expand=True)` yields `Type`/`Level` columns; `MSSubClass.str.split("_", n=1)` keeps only the broad class. **Combine them**: `make + "_" + body_style` gives `make_and_style`
- **Grouped transforms**: `groupby("State")["Income"].transform("mean")` broadcasts each state's mean back onto its own rows (`AverageIncome`); `transform("count") / total` gives a category's population share (`StateFreq`). `transform`, not `agg` — `agg` would collapse the frame to one row per group
- **Leakage rule**: fit a grouped statistic on the *training* split only, then bring it to validation with `merge` — the `AverageClaim` cell splits with `sample(frac=0.5)`, computes the mean per `Coverage` on `df_train`, and left-joins it onto `df_valid`. Computing it on all rows first would leak the validation targets into the features
- New features earn their place by scoring better — that's what exercise 3 measures

### 4. Clustering With K-Means

- **Unsupervised, so it works without a target**: k-means partitions the rows into `n_clusters` groups by repeatedly assigning each row to the nearest centroid and moving the centroids to the mean of their members. The `Cluster` label is then just another feature for a supervised model to split on
- **The lesson's setup**: `X = df.loc[:, ["MedInc", "Latitude", "Longitude"]]` on California housing, `KMeans(n_clusters=6).fit_predict(X)`, then `astype("category")` — the cast matters, an integer cluster id would be read as an ordered quantity. Plot with `relplot(x="Longitude", y="Latitude", hue="Cluster")` to see the clusters are geographic, and `catplot(kind="boxen", x="MedHouseVal", y="Cluster")` to see they separate on the target
- **Scaling is a judgment call, not a default**: k-means measures Euclidean distance, so a feature with large values dominates the clustering. But rescaling is not always right — `Latitude`/`Longitude` must stay as they are (rescaling would distort real distances), doors vs. horsepower must be rescaled (incomparable units), lot area vs. living area is a toss-up. Exercise 1 works through exactly these three cases
- **Standardize before fitting** when you do rescale: exercise 4 uses `(X - X.min(axis=0)) / X.std(axis=0)` over `LotArea`, `TotalBsmtSF`, `FirstFlrSF`, `SecondFlrSF`, `GrLivArea`, then `KMeans(n_clusters=10, n_init=10, random_state=0)` — `n_init` restarts the fit from fresh centroids and keeps the best, and `random_state` makes the run reproducible
- **Two ways to turn clusters into features**: `fit_predict` gives one hard label (each row belongs to exactly one cluster); `fit_transform` gives the distance to *every* centroid, `Centroid_0`…`Centroid_9`, which keeps the graded information "row 12 is 0.3 from cluster 3 and 14.8 from cluster 7" that a single label throws away. The distances scored better in exercise 4
- A cluster column is categorical, so `score_dataset`'s `select_dtypes(["category", "object", "string"])` loop factorizes it automatically — forget that and XGBoost sees raw cluster numbers as an ordered feature

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

The bottom of the ranking — `MoSold`, `LandSlope`, `Threeseasonporch`, `BsmtFinSF2` all at 0.000 — is mostly rare or exceptional conditions that don't describe an average home. Lesson 2's Automobile run on `autos.csv` ranks `curb_weight` (1.517) and `highway_mpg` (0.953) far above the rest, then `length` (0.610), `fuel_system` (0.483) and `stroke` (0.380); `fuel_type` sits at 0.047 and the label-encoded `num_of_doors` at 0.000.

Exercise 3, 5-fold CV √RMSLE with `XGBRegressor` on Ames:

| Feature set | √RMSLE |
| --- | --- |
| Original 78 features | 0.1426 |
| + `LivLotRatio`, `Spaciousness`, `TotalOutsideSF`, `BldgType`×`GrLivArea`, `PorchTypes`, `MSClass`, `MedNhbdArea` | 0.1396 |

Exercise 4, 5-fold CV √RMSLE with `XGBRegressor` on Ames — k-means features added cumulatively to the same 78 original features:

| Feature set | √RMSLE |
| --- | --- |
| Original 78 features (from exercise 3) | 0.1426 |
| + `Cluster` (k=10 over the 5 scaled area features) | 0.1410 |
| + `Centroid_0`…`Centroid_9` (distances from `fit_transform`) | 0.1388 |

The cluster label helps a little; the ten centroid distances help more — about as much as exercise 3's whole hand-built feature set, but derived with one `KMeans` call instead of seven engineered features. Both were scored without touching the hyperparameters (`n_clusters=10, n_init=10, random_state=0` as the exercise specifies).

## Next

Course: [kaggle.com/learn/feature-engineering](https://www.kaggle.com/learn/feature-engineering) — discussion forum [here](https://www.kaggle.com/learn/feature-engineering/discussion).

Four courses now done: `pandas-learn/` (6 sections), `datacleaning-learn/` (5 sections), `intro-machine-learning/` (7 sections), and this one (4 of 6 sections). Next here is **Principal Component Analysis**, then Target Encoding, which close out with the bonus **Feature Engineering for House Prices** notebook — a full submission to the competition. Lesson 1's exercise is still outstanding if you want the course fully ticked off.
