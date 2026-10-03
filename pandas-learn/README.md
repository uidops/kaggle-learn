# pandas-learn

Working notes and exercises for the first section of Kaggle's *Intro to Pandas* micro-course:
**[Creating, Reading and Writing](https://www.kaggle.com/code/residentmario/creating-reading-and-writing)**.

## Files

| File | What it is |
| --- | --- |
| `01-introduction.ipynb` | Lesson walkthrough — DataFrame/Series construction, `index=`, `pd.read_csv()`, `shape`, `head()`, `index_col` |
| `01-exercise.ipynb` | Exercise 1 solutions (5 questions: `fruits`, `fruit_sales`, `ingredients`, `reviews`, `animals`) |
| `cows_and_goats.csv` | Output of question 5 — `animals.to_csv(...)` |
| `winemag-data-130k-v2.csv` | Wine reviews dataset used by `01-introduction.ipynb` — 129,971 rows, 14 cols (13 after `index_col=0`) |
| `winemag-data_first150k.csv` | Wine reviews dataset used by exercise 1, question 4 — 150,930 rows |

Both CSVs come from Kaggle's [Wine Reviews](https://www.kaggle.com/zynicide/wine-reviews) dataset.

## Running the notebooks

```bash
# from the repo root
.venv/bin/jupyter lab pandas-learn/
```

The venv already has pandas (3.0.6), jupyterlab, and nbconvert.

## Concepts covered

- **DataFrame** — a table of labelled rows and columns; built from a dict of lists via `pd.DataFrame({...}, index=[...])`
- **Series** — a single labelled column (a list with an index and an optional `name`)
- **Index** — the row labels; defaults to `0, 1, 2, ...`
- **Reading** — `pd.read_csv(path)`, `.shape`, `.head()`, and `index_col=0` to consume an existing index column
- **Writing** — `df.to_csv("file.csv")`

## Next

Exercise notebook link: [fork the exercise on Kaggle](https://www.kaggle.com/kernels/fork/587970).
