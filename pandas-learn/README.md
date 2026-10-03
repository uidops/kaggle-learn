# pandas-learn

Notes and worked solutions for [Kaggle's *Intro to Pandas* micro-course](https://www.kaggle.com/learn/pandas), run locally instead of on Kaggle.

## Sections

| # | Tutorial | Notebooks |
| --- | --- | --- |
| 1 | [Creating, Reading and Writing](https://www.kaggle.com/code/residentmario/creating-reading-and-writing) | `01-introduction.ipynb`, `01-exercise.ipynb` (5 questions) |
| 2 | [Indexing, Selecting & Assigning](https://www.kaggle.com/code/residentmario/indexing-selecting-assigning) | `02-index.ipynb`, `02-exercise.ipynb` (9 questions) |

Pattern per section: `NN-index`/`NN-introduction` is the lesson walkthrough, `NN-exercise` is the exercise with answers filled in.

| File | What it is |
| --- | --- |
| `01-introduction.ipynb` | Lesson walkthrough — DataFrame/Series construction, `index=`, `pd.read_csv()`, `shape`, `head()`, `index_col` |
| `01-exercise.ipynb` | Exercise 1 solutions: `fruits`, `fruit_sales`, `ingredients`, `reviews`, `animals` |
| `cows_and_goats.csv` | Output of exercise 1 question 5 — `animals.to_csv(...)` |
| `02-index.ipynb` | Lesson walkthrough — attribute access, `[]`, `iloc`/`loc`, slicing, `set_index`, boolean masks, `isin`/`isnull`, column assignment |
| `02-exercise.ipynb` | Exercise 2 solutions, questions 1–9 (selection, label vs. position, conditional filters) |
| `winemag-data-130k-v2.csv` | Wine reviews used by sections 1–2 — 129,971 rows, 14 cols (13 after `index_col=0`) |
| `winemag-data_first150k.csv` | Wine reviews used by exercise 1 question 4 — 150,930 rows |

Both CSVs come from Kaggle's [Wine Reviews](https://www.kaggle.com/zynicide/wine-reviews) dataset and are **gitignored** (≈50 MB each) — download them from Kaggle into `pandas-learn/` if they're missing.

## Running the notebooks

```bash
# from the repo root
.venv/bin/jupyter lab pandas-learn/
```

The venv already has pandas (3.0.6), jupyterlab, and nbconvert. To re-run a notebook headlessly:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute pandas-learn/02-index.ipynb
```

## Concepts covered

### 1. Creating, Reading and Writing

- **DataFrame** — a table of labelled rows and columns; built from a dict of lists via `pd.DataFrame({...}, index=[...])`
- **Series** — a single labelled column (a list with an index and an optional `name`)
- **Index** — the row labels; defaults to `0, 1, 2, ...`
- **Reading** — `pd.read_csv(path)`, `.shape`, `.head()`, and `index_col=0` to consume an existing index column
- **Writing** — `df.to_csv("file.csv")`

### 2. Indexing, Selecting & Assigning

- **Native accessors** — `reviews.country` and `reviews["country"]` both pull one column out as a Series
- **`iloc`** — position-based, row-first/column-second, Python half-open slicing (`0:10` → 10 rows), negatives count from the end
- **`loc`** — label-based, slice bounds are *inclusive* (`loc[0:999]` = 1000 rows where the index is `0..N`) — the main `loc`/`iloc` gotcha
- **`set_index()`** — swap in a better index, e.g. `reviews.set_index("description")`
- **Conditional selection** — a boolean Series (`reviews.country == "Germany"`) passed into `loc`, combined with `&` / `|`, plus `isin([...])` and `isnull()`/`notnull()`
- **Assigning** — new columns from a constant (`df["critic"] = "everyone"`) or an iterable (`range(...)`)

## Next

Exercise forks on Kaggle: [section 1](https://www.kaggle.com/kernels/fork/587970), [section 2](https://www.kaggle.com/kernels/fork/587910).
