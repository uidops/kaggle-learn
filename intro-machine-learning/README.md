# intro-machine-learning

Notes and worked solutions for **[Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)** (Dan Becker) — the third course here, after `pandas-learn/` and `datacleaning-learn/`.

*Builds on: Python. Preparation for: Intermediate Machine Learning, Machine Learning Explainability, Intro to Deep Learning.*

## Sections

| # | Tutorial | Exercise | Steps |
| --- | --- | --- | --- |
| 1 | How Models Work | — | — |
| 2 | [Basic Data Exploration](https://www.kaggle.com/code/dansbecker/basic-data-exploration) | [fork](https://www.kaggle.com/kernels/fork/1258954) | 2 |
| 3 | [Your First Machine Learning Model](https://www.kaggle.com/code/dansbecker/your-first-machine-learning-model) | [fork](https://www.kaggle.com/kernels/fork/1404276) | 4 |
| 4 | [Model Validation](https://www.kaggle.com/code/dansbecker/model-validation) | [fork](https://www.kaggle.com/kernels/fork/1259097) | 4 |
| 5 | [Underfitting and Overfitting](https://www.kaggle.com/code/dansbecker/underfitting-and-overfitting) | [fork](https://www.kaggle.com/kernels/fork/1259126) | 2 |
| 6 | [Random Forests](https://www.kaggle.com/code/dansbecker/random-forests) | [fork](https://www.kaggle.com/kernels/fork/1259186) | 1 |
| 7 | [Machine Learning Competitions](https://www.kaggle.com/code/alexisbcook/machine-learning-competitions) | [fork](https://www.kaggle.com/kernels/fork/1259198) | 3 |

Numbering matches the course, so `NN-exercise.ipynb` pairs directly with lesson `NN`.

| File | What it is |
| --- | --- |
| `02-basic-data-exploration.ipynb` | Lesson walkthrough — `read_csv()`, `describe()` |
| `02-exercise.ipynb` | Exercise 2, steps 1–2 (load Iowa data, review summary stats with `head()`) |
| `03-your-first-machine-learning-model.ipynb` | Lesson walkthrough — pick target `y` and features `X`, `DecisionTreeRegressor`, `fit`/`predict` |
| `03-exercise.ipynb` | Exercise 3, steps 1–4 (specify target, create `X`, fit model, predict) |
| `04-model-validation.ipynb` | Lesson walkthrough — `train_test_split`, `mean_absolute_error` |
| `04-exercise.ipynb` | Exercise 4, steps 1–4 (split, fit, predict on validation, compute MAE) |
| `05-underfitting-and-overfitting.ipynb` | Lesson walkthrough — sweep `max_leaf_nodes`, pick by MAE |
| `05-exercise.ipynb` | Exercise 5, steps 1–2 (compare tree sizes, refit at the best) |
| `06-random-forests.ipynb` | Lesson walkthrough — `RandomForestRegressor` beats the tuned tree |
| `06-exercise.ipynb` | Exercise 6, step 1 (train the forest) |
| `07-exercise.ipynb` | Exercise 7 — full-data fit, predict `test.csv`, write `submission.csv` |

## Datasets

All from Kaggle. Inputs live in `datasets/{dataset-slug}/` and are referenced from the notebooks as `../datasets/…`:

| Used by | Dataset | Path | File | Rows |
| --- | --- | --- | --- | --- |
| Lessons 2–6 | [Melbourne Housing Snapshot](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot) | `datasets/melbourne-housing-snapshot/` | `melb_data.csv` (2.0 MB) | 13,580 |
| Exercises 2–7 | [Home Data for ML Course](https://www.kaggle.com/c/home-data-for-ml-course) — the House Prices competition | `datasets/home-data-for-ml-course/` | `train.csv` (0.44 MB) | 1,460 |
| Exercise 7 only | same competition, test split | `datasets/home-data-for-ml-course/` | `test.csv` | — |

Every input CSV under `datasets/` is gitignored — datasets are downloaded from Kaggle, never committed. `submission.csv` (written by exercise 7) is written next to that notebook and ignored by `intro-machine-learning/*.csv`.

`test.csv` is available from the [competition data page](https://www.kaggle.com/c/home-data-for-ml-course/data) once the rules are accepted.

## Running the notebooks

```bash
# from the repo root
.venv/bin/jupyter lab intro-machine-learning/
```

Notebooks read their CSVs from `../datasets/`, one level up from the notebook.

### Dependencies

| Package | Used by |
| --- | --- |
| `pandas` | every notebook |
| `numpy` | exercises 2 (`np.int64`, `mean().round()`) and 5 (`np.inf`) |
| `datetime` (stdlib) | exercise 2 — `datetime.now().year` for newest-home age |
| `scikit-learn` | lessons 3–6, exercises 3–7 |

Installed in the repo venv: Python 3.14.8, pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1.

## Concepts covered

### 2. Basic Data Exploration

- **You can't model what you haven't looked at.** `df.describe()` gives count / mean / std / min / 25% / 50% / 75% / max per column — 8 numbers that are your first read on a dataset
- **`count` is the tell for missing data**: it counts *non-missing* rows, so a `count` below the row total means `NaN`s are hiding
- `df.head()` (in the exercise) for eyeballing rows after loading — summary stats alone won't show you a mangled column
- Loading is one line: `pd.read_csv(path)`. Point the path at a local file and everything else follows

### 3. Your First Machine Learning Model

- **Prediction target vs features**: `y = data.Price` (one column), `X = data[features]` (the rest). Convention is lowercase `y`, uppercase `X`
- Features are the *inputs*; the model learns their relationship to `y`
- `DecisionTreeRegressor(random_state=1)` → `.fit(X, y)` → `.predict(X)`. The `random_state` pins the random seed so results are reproducible
- Predictions on the training data itself are a smoke test, not a score — the model has already seen every row

### 4. Model Validation

- **Never report accuracy on data the model was trained on** — in-sample predictions look perfect because the model memorised them
- `train_test_split(X, y, random_state=1)` holds back a validation set the model never sees during `fit`
- `mean_absolute_error(y_val, preds)` — the average price the model is off by, in the target's own units (dollars), which makes it readable: *"$29,653 off on average"*
- This is the score every later lesson improves on

### 5. Underfitting and Overfitting

- **Underfitting**: a too-simple model misses real patterns (`max_leaf_nodes=5` → MAE 35,045)
- **Overfitting**: a too-complex model memorises noise and generalises badly (`max_leaf_nodes=500` → MAE 29,454)
- The fix is empirical: sweep candidate values, fit each, keep the one with the lowest validation MAE. Here **100 wins at 27,283**
- The validation set exists precisely so this choice can't be cheated

### 6. Random Forests

- A **forest of decision trees**, each trained on a random slice, averaged at prediction time — variance cancels out where a single tree overfits
- `RandomForestRegressor(random_state=1)` with default settings beat the carefully tuned tree: **27,283 → 21,857**
- Saturated `max_leaf_nodes` tuning was plateauing; changing the *algorithm* was the jump

### 7. Machine Learning Competitions

- Refit on **all** training data once you've finished tuning — the held-back validation rows become training signal
- `test.csv` has no target, so you predict and write `submission.csv` with `index=False` (the competition scores `Id` + `SalePrice`)
- Leaning on the Kaggle-scored feedback loop: submit → compare public/private leaderboard → iterate on features → resubmit

## Results

| Stage | Validation MAE |
| --- | --- |
| Plain `DecisionTreeRegressor` (ex. 4) | 29,653 |
| Tuned `max_leaf_nodes=100` (ex. 5) | 27,283 |
| `RandomForestRegressor` (ex. 6/7) | 21,857 |

## Next

Course: [kaggle.com/learn/intro-to-machine-learning](https://www.kaggle.com/learn/intro-to-machine-learning) — discussion forum [here](https://www.kaggle.com/learn/intro-to-machine-learning/discussion).

The natural follow-on is **[Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning)**, which covers missing values, categorical variables, pipelines, and cross-validation.
