# datacleaning-learn

Notes and worked solutions for Kaggle's **[Data Cleaning](https://www.kaggle.com/learn/data-cleaning)** micro-course — the direct follow-on to `pandas-learn/` (*Builds on: Pandas*).

## Sections

| # | Tutorial | Exercise | Questions | Status |
| --- | --- | --- | --- | --- |
| 1 | [Handling Missing Values](https://www.kaggle.com/code/alexisbcook/handling-missing-values) | [exercise](https://www.kaggle.com/code/alexisbcook/exercise-handling-missing-values) | 6 | ✅ done |
| 2 | [Scaling and Normalization](https://www.kaggle.com/code/alexisbcook/scaling-and-normalization) | [exercise](https://www.kaggle.com/code/alexisbcook/exercise-scaling-and-normalization) | 2 | ✅ done |
| 3 | [Parsing Dates](https://www.kaggle.com/code/alexisbcook/parsing-dates) | [exercise](https://www.kaggle.com/code/alexisbcook/exercise-parsing-dates) | 4 | ✅ done |
| 4 | [Character Encodings](https://www.kaggle.com/code/alexisbcook/character-encodings) | [exercise](https://www.kaggle.com/code/alexisbcook/exercise-character-encodings) | 3 | ✅ done |
| 5 | [Inconsistent Data Entry](https://www.kaggle.com/code/alexisbcook/inconsistent-data-entry) | [exercise](https://www.kaggle.com/code/alexisbcook/exercise-inconsistent-data-entry) | 3 | ✅ done — *course complete* |

18 questions total. Naming follows `pandas-learn/`: `NN-<topic>.ipynb` is the lesson walkthrough, `NN-exercise.ipynb` is the exercise with answers filled in.

| File | What it is |
| --- | --- |
| `01-handling.ipynb` | Lesson walkthrough — `isnull().sum()`, `%` missing via `np.prod(shape)`, `dropna()` vs `dropna(axis=1)`, `fillna()`, `bfill()` |
| `01-exercise.ipynb` | Exercise 1 solutions, questions 1–6 (first look, % missing, why data is missing, drop rows, drop columns, impute) |
| `02-scaling.ipynb` | Lesson walkthrough — `minmax_scaling()` vs `stats.boxcox()`, before/after histograms with `sns.histplot` |
| `02-exercise.ipynb` | Exercise 2 solutions, questions 1–2 (scale `goal`, normalize `pledged`) |
| `03-parsingdates.ipynb` | Lesson walkthrough — `object` vs `datetime64`, `to_datetime(format=...)`, `format='mixed'`, `.dt.day`, histogram sanity check |
| `03-exercise.ipynb` | Exercise 3 solutions, questions 1–4 (dtype check, repair malformed dates then parse, day of month, plot) + optional volcano "Last Known Eruption" bonus |
| `04-character-encodings.ipynb` | Lesson walkthrough — `str`↔`bytes`, `encode`/`decode`, `charset_normalizer.detect()`, read with `encoding=`, write back as UTF-8. **Two cells intentionally raise `UnicodeDecodeError`** (see below) |
| `04-exercise.ipynb` | Exercise 4 solutions, questions 1–3 (`big5-tw`→UTF-8, detect + read `PoliceKillingsUS.csv`, save as `my_file.csv`) |
| `05-inconsistent-data-entry.ipynb` | Lesson walkthrough — `.str.lower()`/`.str.strip()` pre-processing, `rapidfuzz.process.extract` with `token_sort_ratio`, `replace_matches_in_column()` helper |
| `05-exercise.ipynb` | Exercise 5 solutions, questions 1–3 (unique `Graduated from`, strip it, fold `usofa` → `usa`) |
| `Building_Permits.csv` | Exercise 1 dataset — 75 MB *(gitignored)* |
| `NFL Play by Play 2009-2017 (v4).csv` | Lesson 1 dataset — 263 MB *(gitignored)* |

## Datasets

All from Kaggle:

| Used by | Dataset | Local file | Committed? |
| --- | --- | --- | --- |
| Lesson 1 | [NFL Play-by-Play 2009–2016](https://www.kaggle.com/datasets/maxhorowitz/nflplaybyplay2009to2016) | `NFL Play by Play 2009-2017 (v4).csv` (263 MB) | no — gitignored |
| Exercise 1 | [Building Permit Applications](https://www.kaggle.com/datasets/aparnashastry/building-permit-applications-data) | `Building_Permits.csv` (75 MB) | no — gitignored |
| Lesson 2 | *none* — the lesson generates synthetic data with `np.random.exponential` | — | — |
| Exercise 2 | [Kickstarter Projects](https://www.kaggle.com/datasets/kemical/kickstarter-projects) | `ks-projects-201801.csv` | no — gitignored |
| Lesson 3 | [Landslide Events](https://www.kaggle.com/datasets/nasa/landslide-events) | `catalog.csv` (0.4 MB) | no — gitignored |
| Exercise 3 | [Earthquake Database](https://www.kaggle.com/datasets/usgs/earthquake-database) + [Volcanic Eruptions](https://www.kaggle.com/datasets/smithsonian/volcanic-eruptions) | `database.csv` + `volcanic-eruptions-database.csv` (renamed — see below) | no — gitignored |
| Lesson 4 | [Kickstarter Projects](https://www.kaggle.com/datasets/kemical/kickstarter-projects) — the 2016 file | `ks-projects-201612.csv` | no — gitignored |
| Exercise 4 | [Fatal Police Shootings in the US](https://www.kaggle.com/datasets/kwullum/fatal-police-shootings-in-the-us) | `PoliceKillingsUS.csv` (3.2 MB) | no — gitignored |
| Lessons 5 + Exercise 5 | [Pakistan Intellectual Capital](https://www.kaggle.com/datasets/alexisbcook/pakistan-intellectual-capital) | `pakistan_intellectual_capital.csv` (0.2 MB) | no — gitignored |

Every CSV in `datacleaning-learn/` is gitignored by the `datacleaning-learn/*.csv` rule — datasets are downloaded from Kaggle, never committed.

> ⚠️ The earthquake and volcanic datasets both contain a file named `database.csv`. Downloading both side by side silently overwrites the first — so the volcano file is kept here as **`volcanic-eruptions-database.csv`** and `03-exercise.ipynb` reads that name instead of the original `database.csv`.

Gitignored files must be downloaded from Kaggle into `datacleaning-learn/` by hand; the notebooks won't run without them.

## Running the notebooks

```bash
# from the repo root
.venv/bin/jupyter lab datacleaning-learn/
```

Notebooks read their CSVs from this directory (bare filenames, no `../input/` prefix).

### Dependencies

`pandas` and `numpy` were already present. This course adds these packages, **all now installed** in the repo venv:

```bash
.venv/bin/pip install scipy matplotlib seaborn mlxtend charset_normalizer rapidfuzz
```

| Package | Needed by |
| --- | --- |
| `scipy` | Lesson 2 — Box-Cox (`stats.boxcox`) |
| `mlxtend` | Lesson 2 — min-max scaling (`minmax_scaling`) |
| `matplotlib`, `seaborn` | Lessons 2 & 3 — plotting |
| `charset_normalizer` | Lessons 4 & 5 — `detect()` for finding a file's encoding |
| `rapidfuzz` | Lesson 5 — fuzzy matching (`process.extract`, `fuzz.token_sort_ratio`) |

Verified installed: pandas 3.0.6, numpy 2.5.3, scipy 1.18.1, mlxtend 0.25.0, seaborn 0.13.2, matplotlib 3.11.2, charset_normalizer 3.5.2, RapidFuzz 3.14.6.

(`fuzzywuzzy` 0.18.0 is also present — it's what the original Kaggle notebooks import, kept around for comparison, but these notebooks use `rapidfuzz` instead. See *Known issues* below.)

**Jupyter itself had to be installed into the venv too** — it was only in the system Python's user site-packages, so `.venv/bin/jupyter` silently dispatched to the *system* `nbconvert` and the system interpreter (which has `pandas` but none of the packages above). Symptom: `ModuleNotFoundError: No module named 'mlxtend'` even though `.venv/bin/pip list` showed it installed. Now in the venv: `nbconvert`, `nbformat`, `jupyterlab`. The venv's `python3` kernel spec also pointed at a bare `python` (resolving to Homebrew's), and now uses the absolute `.venv/bin/python` path.

## Concepts covered

### 1. Handling Missing Values

- **Counting** — `df.isnull().sum()` per column; total missing ÷ total cells via `np.prod(df.shape)`
- **Why is it missing?** — the key judgement call: *not recorded* (impute it) vs *doesn't exist* (leave `NaN`). Reading the dataset documentation is part of this
- **Dropping** — `dropna()` removes rows with any `NaN` (can empty the frame entirely), `dropna(axis=1)` drops columns instead
- **Filling** — `fillna(0)`, or carry a neighbour value with `bfill()` / `ffill()` when rows have a natural order

### 2. Scaling and Normalization

- **Scaling** changes the *range* (→ 0–1); **normalization** changes the *shape of the distribution* — easy to confuse, different jobs
- Needed when algorithms treat a "1 unit" difference as meaningful (KNN, SVM): 1 yen vs 1 dollar, 1 inch vs 1 pound
- `mlxtend.preprocessing.minmax_scaling` for range, `scipy.stats.boxcox` or a log transform for shape
- Scaling preserves the distribution's shape; normalization does not

### 3. Parsing Dates

- A date column that reads as `object` is really strings — Python doesn't know they're dates
- `pd.to_datetime()` with a format string built from **strftime directives**: `%d` day, `%m` month, `%y` 2-digit year, `%Y` 4-digit year (e.g. `1/17/07` → `%m/%d/%y`)
- Once `datetime64`, the `.dt` accessor gives `.dt.day`, `.dt.month`, etc.
- Plot the parsed column as a sanity check — a flat/uniform distribution means parsing silently failed

### 4. Character Encodings

- **Encoding** maps bytes → characters; reading with the wrong one produces **mojibake** (`æ–‡å—åŒ–`) or replacement characters
- **UTF-8** is the standard — all Python source is UTF-8, and ideally your data too
- `str` vs `bytes`: `s.encode("utf-8")` / `b.decode("utf-8")`; ASCII is English-only and older
- `charset_normalizer.detect(raw_bytes)` guesses the encoding, then pass it to `pd.read_csv(..., encoding=...)`
- Write back out with `df.to_csv(..., encoding="utf-8")`

### 5. Inconsistent Data Entry

- **Pre-processing fixes ~80%**: `.str.strip()` for leading/trailing spaces and `.str.lower()` for case (`' Germany'` / `'germany'`)
- **Fuzzy matching** catches the rest: `rapidfuzz.process.extract(target, choices, limit=10, scorer=rapidfuzz.fuzz.token_sort_ratio)` returns `(value, score, index)` triples; keep those scoring at or above a `min_ratio` and rewrite them
- `token_sort_ratio` ignores word order, so `'usa'` ↔ `'usofa'` scores **75** — enough to fold them together at `min_ratio=70`, but not so loose that unrelated countries get merged (tune it per call)
- A string is "closer" the fewer character changes separate it — automate early rather than hand-fixing thousands of entries

## Known issues with modern versions

- **pandas 3 removed `fillna(method=...)`** — the original lesson 1 calls `fillna(method='bfill', axis=0)`, which raises `TypeError: NDFrame.fillna() got an unexpected keyword argument 'method'`. Use `.bfill(axis=0)` instead. Both `01-*.ipynb` here already use the fixed form (verified re-executed, no errors).
- **`sns.distplot` is deprecated** and will be removed in seaborn 0.14. The original lesson 3 calls it; the notebooks here use the replacement **`sns.histplot(..., kde=True)`** instead, so they run clean on the installed 0.13.2 with no warning.
- **pandas 3 string dtype**: string columns report `str` (`StringDtype`) rather than `object`. Same behavior, newer type name — see the note in `pandas-learn/README.md`.
- **Two cells in `04-character-encodings.ipynb` are *supposed* to raise `UnicodeDecodeError`** — cell 5 (`after.decode("ascii")`) and cell 7 (`pd.read_csv("ks-projects-201612.csv")` with no `encoding=`). They're the lesson's whole point: the first shows ASCII can't hold the euro symbol, the second shows why you must detect the encoding before reading. The Kaggle original does exactly the same. A normal `nbconvert --execute` stops at cell 5; run it with `--ExecutePreprocessor.allow_errors=True` to get through all 11 cells (how this notebook's stored outputs were produced).
- **Lesson 5 uses `rapidfuzz` instead of the original's `fuzzywuzzy`.** `fuzzywuzzy` is unmaintained and `rapidfuzz` is its maintained successor with an API-compatible `process.extract`. One real difference: `rapidfuzz` returns **3-tuples** `(value, score, index)` where `fuzzywuzzy` returns **2-tuples** `(value, score)` — so `[m[0] for m in matches if m[1] >= min_ratio]` works on both, but unpacking `for value, score in ...` does not. Scores are identical for `token_sort_ratio` (`usa`↔`usofa` = 75 in both).

## Next

**Data Cleaning complete** — all 5 lessons, all 18 questions. Course: [kaggle.com/learn/data-cleaning](https://www.kaggle.com/learn/data-cleaning) — discussion forum [here](https://www.kaggle.com/learn/data-cleaning/discussion).

Three courses now done: `pandas-learn/` (6 sections), `datacleaning-learn/` (5 sections), and `intro-machine-learning/` (7 sections).

The Kaggle path branches from here: continue with data work, or follow `intro-machine-learning/` into **Intermediate Machine Learning**. The topics these courses do *not* cover — `merge()`, reshaping (`melt`/`pivot`), categorical dtypes, and duplicate rows — are still open items.
