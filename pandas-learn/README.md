# pandas-learn

Notes and worked solutions for [Kaggle's *Intro to Pandas* micro-course](https://www.kaggle.com/learn/pandas), run locally instead of on Kaggle.

## Sections

| # | Tutorial | Notebooks |
| --- | --- | --- |
| 1 | [Creating, Reading and Writing](https://www.kaggle.com/code/residentmario/creating-reading-and-writing) | `01-introduction.ipynb`, `01-exercise.ipynb` (5 questions) |
| 2 | [Indexing, Selecting & Assigning](https://www.kaggle.com/code/residentmario/indexing-selecting-assigning) | `02-index.ipynb`, `02-exercise.ipynb` (9 questions) |
| 3 | [Summary Functions and Maps](https://www.kaggle.com/code/residentmario/summary-functions-and-maps) | `03-maps.ipynb`, `03-exercise.ipynb` (7 questions) |
| 4 | [Grouping and Sorting](https://www.kaggle.com/code/residentmario/grouping-and-sorting) | `04-grouping.ipynb`, `04-exercise.ipynb` (6 questions) |
| 5 | [Data Types and Missing Values](https://www.kaggle.com/code/residentmario/data-types-and-missing-values) | `05-datatypes.ipynb`, `05-exercise.ipynb` (4 questions) |
| 6 | [Renaming and Combining](https://www.kaggle.com/code/residentmario/renaming-and-combining) | `06-renaming.ipynb`, `06-exercise.ipynb` (4 questions) — *course complete* |

Pattern per section: the `NN-*` lesson notebook is the walkthrough, `NN-exercise` is the exercise with answers filled in.

| File | What it is |
| --- | --- |
| `01-introduction.ipynb` | Lesson walkthrough — DataFrame/Series construction, `index=`, `pd.read_csv()`, `shape`, `head()`, `index_col` |
| `01-exercise.ipynb` | Exercise 1 solutions: `fruits`, `fruit_sales`, `ingredients`, `reviews`, `animals` |
| `cows_and_goats.csv` | Output of exercise 1 question 5 — `animals.to_csv(...)` |
| `02-index.ipynb` | Lesson walkthrough — attribute access, `[]`, `iloc`/`loc`, slicing, `set_index`, boolean masks, `isin`/`isnull`, column assignment |
| `02-exercise.ipynb` | Exercise 2 solutions, questions 1–9 (selection, label vs. position, conditional filters) |
| `03-maps.ipynb` | Lesson walkthrough — `describe()`, `mean()`, `unique()`, `value_counts()`, `map()`, `apply(axis="columns")`, vectorised operators |
| `03-exercise.ipynb` | Exercise 3 solutions, questions 1–7 (median, unique countries, counts, centering, best bargain, descriptor counts, star ratings) |
| `04-grouping.ipynb` | Lesson walkthrough — `groupby()` counts/mins, `apply()` per group, multi-column groups, `agg([len, min, max])`, multi-index, `reset_index()`, `sort_values()`/`sort_index()` |
| `04-exercise.ipynb` | Exercise 4 solutions, questions 1–6 (reviewers written, best rating per price, price extremes per variety, reviewer means, `{country, variety}` MultiIndex counts) |
| `05-datatypes.ipynb` | Lesson walkthrough — `dtype`/`dtypes`, `astype()`, index dtype, `pd.isnull()`, `fillna()`, `replace()` |
| `05-exercise.ipynb` | Exercise 5 solutions, questions 1–4 (points dtype, points as strings, missing prices, `region_1` counts with `fillna`) |
| `06-renaming.ipynb` | Lesson walkthrough — `rename(columns=)`, `rename(index=)`, `rename_axis()`, `pd.concat()`, `set_index()` + `join()` with `lsuffix`/`rsuffix` |
| `06-exercise.ipynb` | Exercise 6 solutions, questions 1–4 (rename locale columns, index name, Reddit products via `concat`, powerlifting meets + lifters via `join`) |
| `winemag-data-130k-v2.csv` | Wine reviews used by sections 1–6 — 129,971 rows, 14 cols (13 after `index_col=0`) |
| `winemag-data_first150k.csv` | Wine reviews used by exercise 1 question 4 — 150,930 rows |

Dataset sources — all from Kaggle:

| Used by | Files | Committed? |
| --- | --- | --- |
| Sections 1–6 | `winemag-data-130k-v2.csv`, `winemag-data_first150k.csv` (≈50 MB each) | no — gitignored |
| Section 6 lesson | `CAvideos.csv`, `GBvideos.csv` — [Trending YouTube](https://www.kaggle.com/datasurvivor/trending-youtube-video-statistics) | no — gitignored (61 MB / 51 MB) |
| Section 6 exercise Q3 | `gaming.csv`, `movies.csv` — [Things on Reddit](https://www.kaggle.com/residentmario/things-on-reddit) | yes (~150 KB combined) |
| Section 6 exercise Q4 | `meets.csv`, `openpowerlifting.csv` — [Powerlifting Database](https://www.kaggle.com/open-powerlifting/powerlifting-database) | `meets.csv` yes (609 KB); `openpowerlifting.csv` no — gitignored (29 MB) |

Gitignored files must be downloaded from Kaggle into `pandas-learn/` by hand; the notebooks won't run without them.

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

### 3. Summary Functions and Maps

- **Summary functions** — `describe()` (type-aware), `mean()`, `median()`, `unique()`, `value_counts()` for a quick read on a column
- **`map()`** — element-wise transform of a Series: `reviews.points.map(lambda p: p - mean)`; returns a new Series, the original is untouched
- **`apply()`** — the same idea over a whole DataFrame; `axis="columns"` passes each row to your function, `axis="index"` passes each column instead
- **Vectorised operators** — `reviews.price - reviews.price.mean()` and `reviews.country + " - " + reviews.region_1` beat `map()`/`apply()` because pandas broadcasts; use `map()`/`apply()` when you need conditional logic

### 4. Grouping and Sorting

- **`groupby()`** — split-apply-combine: `reviews.groupby("points").price.min()` runs the summary function per group; `value_counts()` is just a `groupby()` shortcut
- **`apply()` on groups** — each group arrives as its own DataFrame, e.g. `groupby("winery").apply(lambda df: df.title.iloc[0])`; group by several columns with `groupby(["country", "province"])`
- **`agg()`** — run several summaries at once: `groupby("price").agg([len, min, max])`
- **Multi-index** — grouping by two columns yields an index with multiple levels; `reset_index()` flattens it back to columns, which is usually what you want before sorting
- **Sorting** — `sort_values(by="len")` sorts by value (`ascending=False` for descending, `by=[...]` for several keys), `sort_index()` sorts by the labels instead

### 5. Data Types and Missing Values

- **`dtype` / `dtypes`** — a column's type, or every column's at once; use `astype()` to convert, and remember the index has its own dtype
- **Missing data** — `NaN` entries are always `float64`; select them with `pd.isnull()`/`pd.notnull()`, replace them with `fillna("Unknown")`, or carry the previous value forward with `.ffill()`
- **`replace()`** — swap specific values out, e.g. `taster_twitter_handle.replace("@old", "@new")`

> **pandas 3.0 note:** the tutorial says string columns come out as `object`, but pandas 3.x defaults strings to the dedicated `str` dtype (`StringDtype`). So `reviews.dtypes` shows `country str` rather than `country object`, and `astype("str")` returns `dtype: str` — same behavior, newer type name.

### 6. Renaming and Combining

- **`rename()`** — relabel columns or index entries without touching the data: `rename(columns={"points": "score"})`, `rename(index={0: "firstEntry"})`
- **`rename_axis()`** — changes the *name* of the index/axis itself (`"wines"` for rows, `"fields"` for columns), not the labels; distinct from `set_index()`, which changes which column is the index
- **`pd.concat()`** — stack DataFrames end to end; the default `axis=0` just appends rows. Simplest combine when the two frames share a schema
- **`join()`** — align on a shared key: `set_index("MeetID")` on both sides, then `left.join(right)`, passing `lsuffix`/`rsuffix` when column names collide
- Picking between them: `concat` when the frames are the *same kind* of record, `join` when they describe *different things* keyed together

## Next

Exercise forks on Kaggle: [section 1](https://www.kaggle.com/kernels/fork/587970), [section 2](https://www.kaggle.com/kernels/fork/587910), [section 3](https://www.kaggle.com/kernels/fork/595524), [section 4](https://www.kaggle.com/kernels/fork/598715), [section 5](https://www.kaggle.com/kernels/fork/598826), [section 6](https://www.kaggle.com/kernels/fork/638064).

That's the whole micro-course. Next from here: [Geospatial Analysis](https://www.kaggle.com/learn/geospatial-analysis), [Data Cleaning](https://www.kaggle.com/learn/data-cleaning), or [Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning).
