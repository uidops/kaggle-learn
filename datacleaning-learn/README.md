# datacleaning-learn

Notes and worked solutions for Kaggle's **[Data Cleaning](https://www.kaggle.com/learn/data-cleaning)** micro-course — the direct follow-on to `pandas-learn/` (*Builds on: Pandas*).

## Sections

| # | Tutorial | Exercise | Questions | Status |
| --- | --- | --- | --- | --- |
| 1 | [Handling Missing Values](https://www.kaggle.com/code/alexisbcook/handling-missing-values) | [exercise](https://www.kaggle.com/code/alexisbcook/exercise-handling-missing-values) | 6 | ✅ done |
| 2 | [Scaling and Normalization](https://www.kaggle.com/code/alexisbcook/scaling-and-normalization) | [exercise](https://www.kaggle.com/code/alexisbcook/exercise-scaling-and-normalization) | 2 | ⏳ not started |
| 3 | [Parsing Dates](https://www.kaggle.com/code/alexisbcook/parsing-dates) | [exercise](https://www.kaggle.com/code/alexisbcook/exercise-parsing-dates) | 4 | ⏳ not started |
| 4 | [Character Encodings](https://www.kaggle.com/code/alexisbcook/character-encodings) | [exercise](https://www.kaggle.com/code/alexisbcook/exercise-character-encodings) | 3 | ⏳ not started |
| 5 | [Inconsistent Data Entry](https://www.kaggle.com/code/alexisbcook/inconsistent-data-entry) | [exercise](https://www.kaggle.com/code/alexisbcook/exercise-inconsistent-data-entry) | 3 | ⏳ not started |

18 questions total. Naming follows `pandas-learn/`: `NN-<topic>.ipynb` is the lesson walkthrough, `NN-exercise.ipynb` is the exercise with answers filled in.

| File | What it is |
| --- | --- |
| `01-handling.ipynb` | Lesson walkthrough — `isnull().sum()`, `%` missing via `np.prod(shape)`, `dropna()` vs `dropna(axis=1)`, `fillna()`, `bfill()` |
| `01-exercise.ipynb` | Exercise 1 solutions, questions 1–6 (first look, % missing, why data is missing, drop rows, drop columns, impute) |
| `Building_Permits.csv` | Exercise 1 dataset — 75 MB |
| `NFL Play by Play 2009-2017 (v4).csv` | Lesson 1 dataset — 263 MB |

## Datasets

All from Kaggle:

| Used by | Dataset | Local file | Committed? |
| --- | --- | --- | --- |
| Lesson 1 | [NFL Play-by-Play 2009–2016](https://www.kaggle.com/datasets/maxhorowitz/nflplaybyplay2009to2016) | `NFL Play by Play 2009-2017 (v4).csv` (263 MB) | no — gitignored |
| Exercise 1 | [Building Permit Applications](https://www.kaggle.com/datasets/aparnashastry/building-permit-applications-data) | `Building_Permits.csv` (75 MB) | no — gitignored |
| Lesson 2 | *none* — the lesson generates synthetic data with `np.random.exponential` | — | — |
| Exercise 2 | [Kickstarter Projects](https://www.kaggle.com/datasets/kemical/kickstarter-projects) | `ks-projects-201801.csv` | no — gitignored |
| Lesson 3 | [Landslide Events](https://www.kaggle.com/datasets/nasa/landslide-events) | `catalog.csv` (0.4 MB) | yes, when downloaded |
| Exercise 3 | [Earthquake Database](https://www.kaggle.com/datasets/usgs/earthquake-database) + [Volcanic Eruptions](https://www.kaggle.com/datasets/smithsonian/volcanic-eruptions) | both `database.csv` — **rename on download** (see below) | yes, when downloaded |
| Lesson 4 | [Kickstarter Projects](https://www.kaggle.com/datasets/kemical/kickstarter-projects) — the 2016 file | `ks-projects-201612.csv` | no — gitignored |
| Exercise 4 | [Fatal Police Shootings in the US](https://www.kaggle.com/datasets/kwullum/fatal-police-shootings-in-the-us) | `PoliceKillingsUS.csv` (3.2 MB) | yes, when downloaded |
| Lessons 5 + Exercise 5 | [Pakistan Intellectual Capital](https://www.kaggle.com/datasets/alexisbcook/pakistan-intellectual-capital) | `pakistan_intellectual_capital.csv` (0.2 MB) | yes, when downloaded |

> ⚠️ The earthquake and volcanic datasets both contain a file named `database.csv`. The exercise reads them as two separate frames, so download them into different names (e.g. `earthquake-database.csv`, `volcanic-eruptions.csv`) and adjust the `read_csv` path — otherwise the second download silently overwrites the first.

Gitignored files must be downloaded from Kaggle into `datacleaning-learn/` by hand; the notebooks won't run without them.

## Running the notebooks

```bash
# from the repo root
.venv/bin/jupyter lab datacleaning-learn/
```

Notebooks read their CSVs from this directory (bare filenames, no `../input/` prefix).

### Dependencies

`pandas` and `numpy` were already present. This course adds five packages, **all now installed** in the repo venv:

```bash
.venv/bin/pip install scipy matplotlib seaborn mlxtend fuzzywuzzy python-Levenshtein charset_normalizer
```

| Package | Needed by |
| --- | --- |
| `scipy` | Lesson 2 — Box-Cox (`stats.boxcox`) |
| `mlxtend` | Lesson 2 — min-max scaling (`minmax_scaling`) |
| `matplotlib`, `seaborn` | Lessons 2 & 3 — plotting |
| `charset_normalizer` | Lessons 4 & 5 — `detect()` for finding a file's encoding |
| `fuzzywuzzy` (+ `python-Levenshtein`) | Lesson 5 — fuzzy matching |

Verified installed: pandas 3.0.6, numpy 2.5.3, scipy 1.18.1, mlxtend 0.25.0, seaborn 0.13.2, matplotlib 3.11.2, fuzzywuzzy 0.18.0, charset_normalizer (with `detect`).

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
- **Fuzzy matching** catches the rest: `fuzzywuzzy.process.extract(target, choices, limit=10, scorer=fuzz.token_sort_ratio)`
- A string is "closer" the fewer character changes separate it — automate early rather than hand-fixing thousands of entries

## Known issues with modern versions

- **pandas 3 removed `fillna(method=...)`** — the original lesson 1 calls `fillna(method='bfill', axis=0)`, which raises `TypeError: NDFrame.fillna() got an unexpected keyword argument 'method'`. Use `.bfill(axis=0)` instead. Both `01-*.ipynb` here already use the fixed form (verified re-executed, no errors).
- **`sns.distplot` is deprecated** (lesson 3) and will be removed in seaborn 0.14. It still works on the installed 0.13.2 with a `FutureWarning`; the replacement is `sns.histplot(..., kde=True)`.
- **pandas 3 string dtype**: string columns report `str` (`StringDtype`) rather than `object`. Same behavior, newer type name — see the note in `pandas-learn/README.md`.

## Next

Course: [kaggle.com/learn/data-cleaning](https://www.kaggle.com/learn/data-cleaning) — discussion forum [here](https://www.kaggle.com/learn/data-cleaning/discussion).

After this, the Kaggle path branches: keep going with data work, or start `Intro to ML` (which builds on Python, not Pandas). The topics this course does *not* cover — `merge()`, reshaping (`melt`/`pivot`), categorical dtypes, and duplicate rows — are still open items from `pandas-learn/`.
