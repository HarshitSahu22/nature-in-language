# 🌿 Nature in Language

## What Are We Trying to Find Out?

Across Europe, politicians talk about nature, climate, forests, and the environment all the time. But **saying a word is not the same as treating it as important.** A politician might say "forest" while announcing a routine planting programme — or while declaring a climate emergency. Those are very different situations.

This project asks a simple but powerful question:

> **When politicians mention nature-related words, do they surround them with language that signals urgency and importance — or with neutral, routine language?**

We then go one step further and ask whether the answer differs depending on **who is speaking** — men vs. women, left-wing vs. right-wing politicians, older vs. younger generations, or those in government vs. those in opposition.

> 📌 **In short:** We are not counting how often nature words are mentioned. We are measuring how seriously politicians seem to treat them, based on the words they use around them.

---

## The Data We Are Working With

We use the [ParlaMint corpus](https://www.clarin.eu/parlamint) — a collection of official parliamentary speeches from countries across Europe, translated into English. For each country the data is organised as follows:

| What | What It Contains |
|---|---|
| **Transcript files (.txt)** | Every speech made in parliament, line by line. Each line starts with a unique ID that identifies the exact speech and speaker. |
| **Metadata files (.tsv)** | Information about each speaker — their gender, which party they belong to, whether their party is in government or opposition, and their birth year. |
| **Organised by** | Country → Year → Session (individual sitting of parliament) |

The critical link between the two file types is a shared **unique ID** — for example `ParlaMint-AT_1996-01-15-..._d7e826`. This ID appears at the start of each speech line in the transcript AND as a column in the metadata file, allowing us to know exactly who said what.

> 📂 **File structure:** Each country has one folder. Inside are year subfolders (1996, 1997 …). Inside each year are pairs of files: one transcript (.txt) and one metadata (.tsv) per parliamentary session.

---

## Embedding Methodology — À la Carte (ALC)

This project implements the **À la Carte (ALC) embedding method** from [Khodak et al. (2018)](http://aclweb.org/anthology/P18-1002), which allows us to compute a context-sensitive embedding for any word using only a pre-trained source embedding and a linear transformation matrix.

### Source Embedding: GloVe 840B.300d

We use **GloVe 840B.300d** (Pennington et al., 2014) as our source embedding — the same embedding used in the original ALC paper. It was trained on 840 billion tokens from the Common Crawl and contains vectors for ~2.2 million words, each represented as a 300-dimensional vector.

| Property | Detail |
|---|---|
| Model | GloVe Common Crawl 840B |
| Vocabulary | ~2.2 million words |
| Dimensions | 300 |
| File | `data/embeddings/glove.840B.300d.txt` (~5.3 GB) |
| Source | [Stanford NLP GloVe page](https://nlp.stanford.edu/projects/glove/) |

### Why GloVe over FastText?

The ALC paper (Khodak et al., 2018) was developed and validated specifically with GloVe embeddings. Using GloVe ensures methodological alignment with the paper. Additionally, the Princeton NLP group provides a **pre-trained transform matrix** specifically for GloVe 840B.300d, eliminating the need to learn the matrix from scratch (which requires substantial computation).

### The ALC Transform Matrix

The transform matrix **A** is a 300×300 matrix that maps a raw context-word average into the correct position in the GloVe embedding space. We use the **pre-trained matrix provided by NLPrinceton/ALaCarte**, trained on the full GloVe vocabulary using the Common Crawl corpus.

| Property | Detail |
|---|---|
| Shape | (300, 300) float32 |
| File | `data/alc_transform_glove840B.bin` (352 KB) |
| Source | [NLPrinceton/ALaCarte GitHub repository](https://github.com/NLPrinceton/ALaCarte/tree/master/transform) |
| Format | Raw binary (numpy `fromfile`, float32) |

Because we use the pre-trained Princeton matrix, we **do not need to run `learn_alc_transform.py`**. This script is retained in the repository for reference only.

---

## How We Analyse the Data — Step by Step

### Step 1 — Build a master speaker directory
We read every metadata file across all years and merge them into one complete lookup table for the country. This tells us, for every single speech ever made, who the speaker was and what their demographic attributes are (gender, party alignment, ideology, birth year).

### Step 2 — Assign speakers to groups
Using the demographic information, we sort every speaker into groups. For example under Gender we get Male and Female. Under Ideology we get Left, Centre, and Right. Under Alignment we get Coalition and Opposition. Under Generation we assign birth years to cohorts such as Baby Boomers or Generation X.

### Step 3 — Parse all transcripts and link to speakers
We read every transcript file line by line. For each line we extract the unique ID, look it up in our master directory, and attach the speaker's group labels. The result is a giant table of utterances — millions of rows — each tagged with who said it and in what year.

### Step 4 — Extract context windows around nature words
For every utterance, we scan through the words. Whenever we find one of our ten nature words:

`nature` `climate` `environment` `land` `forest` `forests` `biodiversity` `restoration` `reforestation` `ecology`

…we grab the **10 words immediately before it** and the **10 words immediately after it**. This 20-word window captures the conversational context of that mention. A window of 10 words each side matches the setting used to train the Princeton ALC transform matrix and is the default specified in the ALC paper.

**Example:**

```
Full sentence : "the urgent need to protect the environment is important for land restoration today"
Nature word   : environment (position 6)
Context window: [the, urgent, need, to, protect] [is, important, for, land, restoration]
                ← 5 words before (up to 10)     + 5 words after (up to 10)
```

### Step 5 — Convert words to numbers using GloVe
We use GloVe 840B.300d to represent every context word as a 300-dimensional vector. Words used in similar contexts end up with similar vectors — so "forest" and "woodland" would be close together, while "forest" and "economy" would be far apart.

### Step 6 — Compute the ALC embedding for each nature word per group per year
For a given group (e.g. Female speakers) and year (e.g. 2005), we take all the context windows collected around "climate":

1. For each occurrence, **sum** the GloVe vectors of all context words in the window
2. Average those summed vectors across all occurrences → one raw context vector
3. Multiply by the transform matrix **A**: `alc_embedding = raw_context_avg @ A.T`

The sum (not mean) within each window matches the convention used when training the Princeton matrix.

```
All context windows for "climate" (Female, 2005)
        ↓
For each window: sum the GloVe vectors of the context words → one vector per window
        ↓
Average all window vectors → one raw context vector
        ↓
Multiply by Princeton transform matrix A → corrected ALC embedding
        ↓
This is the ALC embedding: "how Female speakers used climate in 2005"
```

### Step 7 — Measure similarity to importance words
We also get GloVe vectors (static, not ALC-transformed) for four importance words:

`important` `importance` `significant` `meaningful`

We then measure the **cosine similarity** between the nature word's ALC embedding and each importance word's static GloVe vector.

- Score **close to 1.0** → the nature word was used in contexts very similar to how importance words are used — suggesting the topic was framed as urgent and critical
- Score **close to 0** → the contexts were unrelated — the topic was discussed in neutral or routine language

> **Why use static GloVe vectors for importance words?** Because the ALC transform maps context averages into the same space as the original GloVe vectors. Comparing an ALC embedding to a static GloVe vector is the intended use of the method — both live in the same vector space.

### Step 8 — Save results and repeat for every group
We save a table for each group and each year showing the similarity scores — **10 nature words × 4 importance words = 40 scores per table**. We do this separately for every group within every dimension, so you can compare Male vs Female, or Left vs Right, side by side.

---

## The Four Dimensions We Compare

Rather than looking at all politicians together, we split speeches by four different characteristics. This lets us ask whether different types of politicians frame nature differently.

| Dimension | Groups |
|---|---|
| **Gender** | Male \| Female |
| **Political Alignment** | Coalition (parties in government) \| Opposition (parties not in government) |
| **Political Ideology** | Left \| Centre \| Right |
| **Generation** | Silent Generation (1928–1945) \| Baby Boomers (1946–1964) \| Generation X (1965–1980) \| Millennials (1981–1996) \| Generation Z (1997–2012) |

---

## How to Read the Output Files

For each group (e.g. Left-wing speakers) and each year, the analysis produces one CSV file — a spreadsheet you can open in Excel. It looks like this:

| Nature word ↓ | important | importance | significant | meaningful |
|---|---|---|---|---|
| climate | 0.61 | 0.58 | 0.63 | 0.55 |
| environment | 0.57 | 0.55 | 0.59 | 0.51 |
| forest | 0.35 | 0.33 | 0.41 | 0.31 |
| nature | 0.42 | 0.40 | 0.51 | 0.38 |
| biodiversity | 0.45 | 0.43 | 0.49 | 0.40 |
| restoration | 0.38 | 0.36 | 0.44 | 0.35 |
| reforestation | 0.29 | 0.27 | 0.33 | 0.26 |
| land | 0.33 | 0.30 | 0.40 | 0.29 |
| forests | 0.34 | 0.32 | 0.39 | 0.30 |
| ecology | 0.40 | 0.38 | 0.46 | 0.37 |

**Each number is a score between 0 and 1.** The closer to 1, the more the nature word was spoken in contexts that resemble how importance words are used.

In the example above:
- **climate at 0.61** → politicians discussing climate used surrounding language very similar to what they use when discussing important matters — framed as urgent
- **reforestation at 0.29** → discussed in more neutral, technical language — framed as routine

---

## Where the Output Files Are Saved

The results are saved in a folder structure that mirrors the four dimensions. For example, for Finland (FI):

```
results/
├── transcript/
│   └── FI/
│       ├── 1996_nature_vs_importance.csv
│       ├── 1997_nature_vs_importance.csv
│       ├── contexts_by_term_year.csv
│       └── alc_embeddings.pkl
└── meta_transcript/
    └── FI/
        ├── gender/
        │   ├── Male/
        │   │   ├── 1996_nature_vs_importance.csv
        │   │   ├── contexts_by_term_year.csv
        │   │   └── alc_embeddings.pkl
        │   └── Female/
        │       └── (same files)
        ├── alignment/
        │   ├── Coalition/ ...
        │   └── Opposition/ ...
        ├── ideology/
        │   ├── Left/ ...
        │   ├── Centre/ ...
        │   └── Right/ ...
        └── generation/
            ├── Silent_Generation/ ...
            ├── Baby_Boomers/ ...
            ├── Generation_X/ ...
            ├── Millennials/ ...
            └── Generation_Z/ ...
```

Each group folder also contains two additional files for researchers who want to dig deeper:
- `contexts_by_term_year.csv` — the raw context windows that were collected
- `alc_embeddings.pkl` — the numerical vectors used in the similarity calculation

---

## A Simple Analogy

> *Imagine you are reading thousands of news articles and you want to know whether journalists treat "forest" as an important topic or a routine one.*
>
> *You notice that when journalists write about "crisis", "urgent", "must act", "critical" — those words tend to cluster together. And when they write about "important" or "significant", those same urgent words appear nearby.*
>
> *Now you look at what words surround "forest". If the surrounding words are also "crisis", "urgent", "protect", "must" — then forest is being treated as important. If the surrounding words are "hectares", "programme", "allocated", "schedule" — then it is being treated as administrative routine.*
>
> **This analysis does exactly that — but at scale, across millions of parliamentary speeches, across many countries and many years.**

---

## Key Terms Explained Simply

| Term | What It Means in Plain Language |
|---|---|
| **GloVe 840B.300d** | A pre-trained word embedding model trained on 840 billion tokens from the web. It represents every word as a list of 300 numbers. Used in the original ALC paper. |
| **Word vector** | A list of 300 numbers that represents a word's meaning. Words with similar meanings have similar vectors. |
| **Context window** | The 10 words before and 10 words after a target word (20 words total). Captures the conversational environment of a word. This size matches the ALC paper and the pre-trained transform matrix. |
| **ALC embedding** | The average of context-word sum vectors, multiplied by the Princeton transform matrix **A**. Maps "how a word was used" into the GloVe vector space. (Khodak et al., 2018) |
| **Transform matrix A** | A 300×300 matrix pre-trained by Princeton NLP on the full GloVe vocabulary. Corrects for frequency bias and aligns context averages with the source embedding space. |
| **Cosine similarity** | A score from 0 to 1 measuring how similar two word vectors are. 1 = identical usage pattern, 0 = completely unrelated. |
| **Utterance** | A single speech or statement by one speaker in one session of parliament. |
| **Dimension** | One of the four ways we split speakers: Gender, Alignment, Ideology, or Generation. |

---

## Important Limitations to Keep in Mind

- **This method measures framing, not intent.** A high similarity score means the language pattern resembles importance — it does not prove the politician personally cares about the topic.
- **Speeches are translated to English.** Some nuance from the original language may be lost in translation.
- **Party ideology labels come from the ParlaMint dataset.** They reflect how parties positioned themselves at the time and may not capture all political nuance.
- **Some speakers appear across many years.** Their demographic attributes (e.g. party alignment) are taken from their first appearance in the dataset and assumed stable over time.
- **Nature terms and importance terms were chosen by the research team.** Different word choices would produce different results.
- **The transform matrix was trained on general web text (Common Crawl).** It is not specific to parliamentary language, which may affect the precision of the ALC embeddings.

---

## 🔧 Setup Instructions

Follow the steps below to set up the environment and run the project locally.

### 1. Clone the repository
```bash
git clone https://github.com/HarshitSahu22/nature-in-language.git
cd nature-in-language
```

### 2. (Recommended) Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download required data files

The two large data files are not stored in the repository. Download them manually:

#### GloVe 840B.300d embeddings (~2 GB download, ~5.3 GB extracted)
```bash
cd data/embeddings
curl -LO https://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip
cd ../..
```

#### Princeton ALC transform matrix (352 KB download)
```bash
curl -L -o data/alc_transform_glove840B.bin \
  https://github.com/NLPrinceton/ALaCarte/raw/master/transform/840B.300d.bin
```

### 5. Run the analysis
```bash
python3 run_all_countries.py
```

> ⚠️ Loading GloVe 840B takes approximately 5–10 minutes and ~2.6 GB of RAM. Close other memory-intensive applications before running. The per-country analysis is processed sequentially and is much lighter.

---

## 📁 Folder Descriptions

### **`/data/`**
Main folder containing embeddings, the transform matrix, raw corpora, and all output results.

#### **`/data/embeddings/glove.840B.300d.txt`**
Pre-trained GloVe word embeddings (840B token Common Crawl, 300 dimensions). This is the source embedding used in all ALC computations. ~2.2 million words, ~5.3 GB.

#### **`/data/alc_transform_glove840B.bin`**
The pre-trained ALC transform matrix (300×300, float32 binary). Provided by [NLPrinceton/ALaCarte](https://github.com/NLPrinceton/ALaCarte). Applied as `raw_context_avg @ A.T` to produce ALC embeddings.

#### **`/data/raw/`**
Machine-translated English versions of each country's ParlaMint parliamentary proceedings, organised by country and year.

#### **`/data/results/transcript/`**
Country-level analysis results (all speakers combined, split by year only).

#### **`/data/results/meta_transcript/`**
Demographic-split analysis results (broken down by gender, alignment, ideology, and generation within each country and year).

---

## 📄 File Descriptions

### **`run_all_countries.py`**
The main analysis script. Loops over all 29+ ParlaMint country corpora and runs two analyses for each:
1. **Country-level transcript analysis** — all speakers combined, by year
2. **Demographic-split analysis** — broken down by gender, alignment, ideology, and generation

**Key configuration (top of file):**
```python
EMBED_PATH  = "data/embeddings/glove.840B.300d.txt"   # GloVe source embedding
MATRIX_PATH = "data/alc_transform_glove840B.bin"       # Princeton pre-trained matrix
WINDOW      = 10                                        # Context words each side (matches ALC paper)
```

**For each nature term per group per year, it:**
- Extracts ±10 word context windows from speeches
- Sums GloVe vectors within each window
- Averages across all occurrences
- Applies the transform matrix A
- Computes cosine similarity against 4 importance terms
- Saves a 10×4 CSV table

---

### **`metaData.ipynb`**
Exploratory notebook for understanding the ParlaMint metadata structure — speaker counts, demographic breakdowns, and coverage across countries and years.

---

### **`/data/results/transcript/{country}/contexts_by_term_year.csv`**
Raw context windows (±10 words) around each nature term, organised by term and year. Useful for qualitative inspection of what language surrounds each term.

---

### **`/data/results/transcript/{country}/alc_embeddings.pkl`**
Computed ALC embeddings for each nature term by year.
**Format:** `{term: {year: numpy array (300,)}}`

---

### **`/data/results/transcript/{country}/{year}_nature_vs_importance.csv`**
The primary output — cosine similarity scores between each nature term's ALC embedding and each importance term's static GloVe vector.
**Format:** 10 rows (nature terms) × 4 columns (importance terms)

---

### **`/requirements.txt`**
Lists all Python packages required to run the project with pinned versions for reproducibility.

---

## References

- Khodak, M., Saunshi, N., Liang, Y., Ma, T., Stewart, B., & Arora, S. (2018). *A La Carte Embedding: Cheap but Effective Induction of Semantic Feature Vectors.* Proceedings of ACL 2018. http://aclweb.org/anthology/P18-1002
- Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global Vectors for Word Representation.* EMNLP 2014.
- NLPrinceton/ALaCarte GitHub repository: https://github.com/NLPrinceton/ALaCarte
- ParlaMint corpus: https://www.clarin.eu/parlamint

---

*For questions about the methodology, refer to the script `run_all_countries.py` and the paper cited above.*
