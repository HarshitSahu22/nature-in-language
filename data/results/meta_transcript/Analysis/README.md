# 📊 Analysis — Parliamentary Nature Framing Study

> **Location:** `.../results/meta_transcript/analysis/`
> **Project:** KelloggXParlaMint — Semantic framing of nature & environment in European parliamentary speech
> **Pipeline stage:** Final outputs — averages, visualisations, and time-series analyses across demographic dimensions

---

## What This Folder Contains

This folder holds all computed outputs from the **meta-dimension averaging pipeline** (`meta_average_analysis.ipynb`). The pipeline takes the per-country, per-year cosine similarity scores produced by `natureImportance_byDimension.ipynb` and aggregates, compares, and visualises them across four demographic dimensions.

---

## Folder Structure

```
analysis/
│
├── README.md                          ← this file
│
├── meta_averages_by_group.xlsx        ← master Excel workbook (11 sheets)
│
├── Heatmaps/
│    ├── heatmap_alignment.png              ┐
│    ├── heatmap_gender.png                 │  Core heatmaps
│    ├── heatmap_generation.png             │  (side-by-side groups per dimension)
│    ├── heatmap_ideology.png               ┘
│
├── LineGraphs/
│    ├── linegraph_alignment.png            ┐
│    ├── linegraph_gender.png               │  Core line graphs
│    ├── linegraph_generation.png           │  (one line per country, per group)
│    ├── linegraph_ideology.png             ┘
│
├── low_complexity/                    ← Analyses 1–3
│   ├── loess_divergence_alignment.png
│   ├── loess_divergence_gender.png
│   ├── loess_divergence_generation.png
│   ├── loess_divergence_ideology.png
│   ├── gap_series_all_dimensions.png
│
└── medium_complexity/                 ← Analyses 4–6
    ├── rolling_corr_alignment.png
    ├── rolling_corr_gender.png
    ├── rolling_corr_generation.png
    ├── rolling_corr_ideology.png
    ├── variance_decomp_alignment.png
    ├── variance_decomp_gender.png
    ├── variance_decomp_generation.png
    ├── variance_decomp_ideology.png
    ├── rank_stability_alignment.png
    ├── rank_stability_gender.png
    ├── rank_stability_generation.png
    └── rank_stability_ideology.png
```

---

## The Excel Workbook — `meta_averages_by_group.xlsx`

**11 sheets**, one per demographic group. Each sheet is a pivot table:

- **Rows** → Years (full global range, e.g. 1996–2024)
- **Columns** → Country codes (all countries in the dataset, A–Z)
- **Cell value** → Mean cosine similarity score for that group × country × year, averaged across all 10 nature terms and 4 importance terms (40 values per CSV)
- **Empty cells** → No data available for that group in that country-year combination

| Sheet name | Dimension |
|---|---|
| `Coalition` | Alignment |
| `Opposition` | Alignment |
| `Male` | Gender |
| `Female` | Gender |
| `Silent_Generation` | Generation |
| `Baby_Boomers` | Generation |
| `Generation_X` | Generation |
| `Millennials` | Generation |
| `Generation_Z` | Generation |
| `Left` | Ideology |
| `Centre` | Ideology |
| `Right` | Ideology |

> **What the score means:** A value of `0.45` means that when politicians in that group talked about nature-related words (climate, forest, biodiversity etc.), the surrounding language was 45% similar — in vector space — to how importance/significance language is typically used. Higher = nature framed as more urgent and important.

---

## Core Visualisations

### Heatmaps (`heatmap_{dimension}.png`)

- Groups in the dimension shown **side by side** for direct comparison
- **Rows** = countries, **Columns** = years
- **Colour scale** (`BuGn`): light = lower similarity, dark green = higher similarity
- **White cells** = no data for that country-year
- Shared colour scale within each dimension so panels are directly comparable

### Line Graphs (`linegraph_{dimension}.png`)

- Groups shown **side by side**
- **One line per country**, x-axis = year, y-axis = mean similarity score
- Missing years produce gaps in the line (no interpolation)
- Legend outside right panel

---

## Low Complexity Analyses (`low_complexity/`)

### Analysis 1 & 3 — LOESS Trends + Divergence (`loess_divergence_{dim}.png`)

Two-row figure per dimension:

- **Top row:** Raw scatter points with LOESS-smoothed trend lines (bandwidth = 0.35). Separates long-run signal from year-to-year noise.
- **Bottom row:** Each country's score **minus the dimension mean** for that year. Values above zero = that group in that country frames nature as *more* important than the dimension average. The black dashed line at zero is the reference.

### Analysis 2 — Gap Series (`gap_series_all_dimensions.png`)

Four-panel figure. Each panel shows `Score(Group A) − Score(Group B)` per country over time for one dimension:

| Dimension | Group A | Group B |
|---|---|---|
| Alignment | Coalition | Opposition |
| Gender | Male | Female |
| Ideology | Left | Right |
| Generation | Millennials | Baby Boomers |

- Positive gap → Group A frames nature as more important that year
- **Sign flips** (line crossing zero) = the leading group switched
- Blue shaded band = cross-country mean gap

## Medium Complexity Analyses (`medium_complexity/`)

### Analysis 5 — Rolling 3-Year Correlation (`rolling_corr_{dim}.png`)

**Question:** Do the two groups in a dimension move in sync, or are they diverging?

- **Left panel (heatmap):** Rolling Pearson correlation between Group A and Group B for each country, computed over a 3-year window. Colormap: `RdYlGn` — green = groups moving together, red = groups diverging, grey = insufficient data
- **Right panel (line):** Cross-country mean correlation over time with shaded band showing the range across all countries
- Reference lines at r = 0 (no relationship), r = +0.5 (moderate sync), r = −0.5 (moderate divergence)

### Analysis 6 — Variance Decomposition (`variance_decomp_{dim}.png`)

**Question:** Is variation in scores driven more by *which country* a politician is from, or *which group* they belong to?

- **Between-country variance** (blue): How much do countries differ from each other within a group in a given year?
- **Between-group variance** (coral): How much do groups differ from each other within a country in a given year?
- **Left panel:** Raw variance values over time
- **Right panel:** Stacked area chart showing percentage share. If the coral band grows → groups are polarising. If blue dominates → geography explains more than demographics.

### Analysis 7 — Rank Stability / Bump Charts (`rank_stability_{dim}.png`)

**Question:** Does the same group consistently frame nature as most important, or does leadership change?

- **Left panel (bump chart):** Groups ranked 1–N by mean cross-country score each year. Rank 1 at top = highest framing score. Crossing lines = rank swap.
- **Right panel (rank-change heatmap):** Year-on-year change in rank per group. Green = rose in rank that year, red = dropped. Colormap: `RdYlGn_r`.

---

## How Scores Are Computed (Summary)

```
Raw data:   {country}/{dimension}/{group}/{year}_nature_vs_importance.csv
            → 10 × 4 table (10 nature terms × 4 importance terms)
            → 40 cosine similarity values per file

Step 1:     Flatten the 40 values, compute nanmean → one scalar per file
Step 2:     Collect all scalars into a flat table (Country, Dimension, Group, Year, MeanValue)
Step 3:     Pivot to Year × Country per group → stored in Excel + used for all plots
```

---

## Key Parameters

| Parameter | Value | Set in |
|---|---|---|
| Nature terms | nature, climate, environment, land, forest, forests, biodiversity, restoration, reforestation, ecology | `natureImportance_byDimension.ipynb` |
| Importance terms | important, importance, significant, meaningful | `natureImportance_byDimension.ipynb` |
| Context window | ±5 words | `natureImportance_byDimension.ipynb` |
| Embedding model | FastText `cc.en.300.vec` (300-dim) | `natureImportance_byDimension.ipynb` |
| LOESS bandwidth | frac = 0.35 | `meta_average_analysis.ipynb` |
| Rolling window | 3 years | `meta_average_analysis.ipynb` |

---

## Regenerating the Outputs

Run all cells in `meta_average_analysis.ipynb` in order. The notebook is self-contained — it reads from `meta_transcript/{country}/{dimension}/{group}/` and writes everything to this `analysis/` folder.

```
Cell 0  → Config & paths
Cell 1  → Collect all CSV data
Cell 2  → Coverage & range diagnostics
Cell 3  → Build pivot tables + save Excel
Cell 4  → Pivot spot-checks
Cell 5  → Plot helper functions
Cell 6  → Core heatmaps
Cell 7  → Core line graphs
Cell 8  → Final summary (core outputs)
Cell 9  → Low complexity analyses (1–3)
Cell 10 → Medium analysis 4: Rolling correlation
Cell 11 → Medium analysis 5: Variance decomposition
Cell 12 → Medium analysis 6: Rank stability
Cell 13 → Complete output summary
```

---

## Notes & Caveats

- Scores represent **framing**, not frequency. A high score means a group used nature words in contexts semantically similar to importance language — not that they mentioned nature more often.
- Some country-group-year combinations have no data (white cells / gaps) because that demographic group had no recorded speeches in ParlaMint for that year.
- Speeches are machine-translated to English. Some semantic nuance from source languages may affect scores.
- Generation labels are assigned by birth year from the metadata. Speakers appearing across many years retain the same generation label throughout.
- The `Generation_Z` sheet will be sparse — very few parliamentarians born after 1997 appear in the corpus.