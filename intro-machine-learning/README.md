# intro-machine-learning

Notes and worked solutions for **[Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)** (Dan Becker) — the third course here, after `pandas-learn/` and `datacleaning-learn/`.

*Builds on: Python. Preparation for: Intermediate Machine Learning, Machine Learning Explainability, Intro to Deep Learning.*

## Sections

| # | Tutorial | Exercise | Steps | Status |
| --- | --- | --- | --- | --- |
| 1 | How Models Work | — | — | 📖 no code — prose only, so no notebook |
| 2 | [Basic Data Exploration](https://www.kaggle.com/code/dansbecker/basic-data-exploration) | [fork](https://www.kaggle.com/kernels/fork/1258954) | 2 | ✅ done |
| 3 | [Your First Machine Learning Model](https://www.kaggle.com/code/dansbecker/your-first-machine-learning-model) | [fork](https://www.kaggle.com/kernels/fork/1404276) | 4 | ✅ done |
| 4 | [Model Validation](https://www.kaggle.com/code/dansbecker/model-validation) | [fork](https://www.kaggle.com/kernels/fork/1259097) | 4 | ✅ done |
| 5 | [Underfitting and Overfitting](https://www.kaggle.com/code/dansbecker/underfitting-and-overfitting) | [fork](https://www.kaggle.com/kernels/fork/1259126) | 2 | ✅ done |
| 6 | [Random Forests](https://www.kaggle.com/code/dansbecker/random-forests) | [fork](https://www.kaggle.com/kernels/fork/1259186) | 1 | ✅ done |
| 7 | [Machine Learning Competitions](https://www.kaggle.com/code/alexisbcook/machine-learning-competitions) | [fork](https://www.kaggle.com/kernels/fork/1259198) | 3 * | ⚠️ exercise only — 2 cells need `test.csv` |

Section 7's exercise has no `Step N` markers like the others — the count is its three instructions (fit on all data, predict the test set, generate the submission).

**Numbering starts at 02.** Lesson 1 (*How Models Work*) has no code icon on the course page — it's a conceptual intro, so there's no notebook for it. Numbering here matches the course, not the file count.

Unlike the other two courses, exercises use **Steps** rather than numbered questions, so `NN-exercise.ipynb` pairs directly with lesson `NN`.

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
| `07-exercise.ipynb` | Exercise 7 — full-data fit, predict `test.csv`, write `submission.csv` (last 2 cells pending) |

> ⚠️ There is **no `07` lesson notebook** here — the lesson 7 tutorial exists on Kaggle but wasn't saved locally. Add `07-machine-learning-competitions.ipynb` if you want the pair complete.

## Datasets

All from Kaggle:

| Used by | Dataset | Local file | Rows | Committed? |
| --- | --- | --- | --- | --- |
| Lessons 2–6 | [Melbourne Housing Snapshot](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot) | `melb_data.csv` (2.0 MB) | 13,580 | no — gitignored |
| Exercises 2–7 | [Home Data for ML Course](https://www.kaggle.com/c/home-data-for-ml-course) — the House Prices competition | `train.csv` (0.44 MB) | 1,460 | no — gitignored |
| Exercise 7 only | same competition, test split | **`test.csv` — missing** | — | no — gitignored |

Every CSV in `intro-machine-learning/` is ignored by the `intro-machine-learning/*.csv` rule — datasets are downloaded from Kaggle, never committed. `submission.csv` (written by exercise 7) is covered by the same rule.

`test.csv` is **not in the repo**. Get it from the [competition data page](https://www.kaggle.com/c/home-data-for-ml-course/data) (accept the rules first), drop it in `intro-machine-learning/`, and the last two cells of `07-exercise.ipynb` will run.

## Running the notebooks

```bash
# from the repo root
.venv/bin/jupyter lab intro-machine-learning/
```

Notebooks read their CSVs from this directory (bare filenames, no `../input/` prefix).

### Dependencies

**No new packages needed** — everything this course uses was already in the venv:

| Package | Used by |
| --- | --- |
| `pandas` | every notebook |
| `numpy` | exercises 2 (`np.int64`, `mean().round()`) and 5 (`np.inf`) |
| `datetime` (stdlib) | exercise 2 — `datetime.now().year` for newest-home age |
| `scikit-learn` | lessons 3–6, exercises 3–7 |

Verified installed: Python 3.14.8, pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1.

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

## Known issues

- **`07-exercise.ipynb` has 2 unexecuted cells** (the only gap in an otherwise fully-executed directory — 43/45 cells). Cell 6 does `pd.read_csv('test.csv')` → `FileNotFoundError`, and cell 9 then fails with `NameError: test_data is not defined` as a knock-on. Everything else in the notebook runs: the RandomForest fits and prints `Validation MAE for Random Forest Model: 21,857`. Download `test.csv` (above) and both cells run.
- **No `07` lesson notebook** — see the note in *Files*.
- **No Kaggle API credentials on this machine** (`~/.kaggle/kaggle.json` absent, `kaggle` module not installed), so `test.csv` can't be fetched automatically.

## Results

Verified by re-executing every notebook headlessly (exit 0, no error cells):

| Stage | Validation MAE |
| --- | --- |
| Plain `DecisionTreeRegressor` (ex. 4) | 29,653 |
| Tuned `max_leaf_nodes=100` (ex. 5) | 27,283 |
| `RandomForestRegressor` (ex. 6/7) | 21,857 |

These match the canonical Kaggle answers.

## Next

Course: [kaggle.com/learn/intro-to-machine-learning](https://www.kaggle.com/learn/intro-to-machine-learning) — discussion forum [here](https://www.kaggle.com/learn/intro-to-machine-learning/discussion).

The natural follow-on is **[Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning)**, which covers the things this course deliberately skipped: missing values, categorical variables, pipelines, and cross-validation — two of which (`merge()`, categorical dtypes) are still open from `pandas-learn/`.
