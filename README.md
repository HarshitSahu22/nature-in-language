# 🌿 Nature in Language

## What Are We Trying to Find Out?

Across Europe, politicians talk about nature, climate, forests, and the environment all the time. But **saying a word is not the same as treating it as important.** A politician might say "forest" while announcing a routine planting programme — or while declaring a climate emergency. Those are very different situations.

This project asks a simple but powerful question:

> **When politicians mention nature-related words, do they surround them with language that signals urgency and importance — or with neutral, routine language?**

We then go one step further and ask whether the answer differs depending on **who is speaking** — men vs. women, left-wing vs. right-wing politicians, older vs. younger generations, or those in government vs. those in opposition.

> 📌 **In short:** We are not counting how often nature words are mentioned. We are measuring how seriously politicians seem to treat them, based on the words they use around them.

---

## The Data We Are Working With
We use **word2vec** and other embedding models (e.g., FastText, BERT) on data provided by the [ParlaMint corpus](https://www.clarin.eu/parlamint). It is a collection of official parliamentary speeches from countries across Europe, translated into English. For each country the data is organised as follows:

| What | What It Contains |
|---|---|
| **Transcript files (.txt)** | Every speech made in parliament, line by line. Each line starts with a unique ID that identifies the exact speech and speaker. |
| **Metadata files (.tsv)** | Information about each speaker — their gender, which party they belong to, whether their party is in government or opposition, and their birth year. |
| **Organised by** | Country → Year → Session (individual sitting of parliament) |

The critical link between the two file types is a shared **unique ID** — for example `ParlaMint-AT_1996-01-15-..._d7e826`. This ID appears at the start of each speech line in the transcript AND as a column in the metadata file, allowing us to know exactly who said what.

> 📂 **File structure:** Each country has one folder. Inside are year subfolders (1996, 1997 …). Inside each year are pairs of files: one transcript (.txt) and one metadata (.tsv) per parliamentary session.

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

…we grab the **five words immediately before it** and the **five words immediately after it**. This 10-word window captures the conversational context of that mention.

**Example:**

```
Full sentence : "the environment is important for land restoration today"
Nature word   : environment (position 2)
Context window: [the] [is] [important] [for] [land]
                ← 1 word before + 4 words after (excludes "environment" itself)
```

### Step 5 — Convert words to numbers using an AI word model
We use a pre-trained AI model called **FastText**, which has learned from billions of text documents. It represents every word as a list of 300 numbers (called a **vector**) that captures its meaning. Words used in similar contexts end up with similar vectors — so "forest" and "woodland" would be close together, while "forest" and "economy" would be far apart.

### Step 6 — Compute the "average meaning" of each nature word per group per year
For a given group (e.g. Female speakers) and year (e.g. 2005), we take all the context windows collected around "climate". We convert each context word to its FastText vector, average them to get one vector per window, then average all those vectors together. The result is a single number-list that represents **how "climate" was talked about by Female speakers in 2005**.

```
All context windows for "climate" (Female, 2005)
        ↓
Convert each word to a 300-number vector
        ↓
Average the vectors in each window → one vector per window
        ↓
Average all window vectors → one final vector
        ↓
This is the ALC embedding: "how Female speakers used climate in 2005"
```

### Step 7 — Measure similarity to importance words
We also get FastText vectors for four importance words:

`important` `importance` `significant` `meaningful`

We then measure the **mathematical angle** between the nature word's context vector and each importance word's vector.

- Score **close to 1.0** → the nature word was used in contexts very similar to how importance words are used — suggesting the topic was framed as urgent and critical
- Score **close to 0** → the contexts were unrelated — the topic was discussed in neutral or routine language

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
└── meta_transcript/
    └── FI/
        ├── gender/
        │   ├── Male/
        │   │   ├── 1996_nature_vs_importance.csv
        │   │   ├── 1997_nature_vs_importance.csv
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
| **FastText** | An AI model trained on billions of words. It has learned which words tend to appear together, and uses that knowledge to represent words as numbers. |
| **Word vector** | A list of 300 numbers that represents a word's meaning. Words with similar meanings have similar vectors. |
| **Context window** | The 5 words before and 5 words after a target word. Captures the conversational environment of a word. |
| **ALC embedding** | The average vector of all context windows for a word in a given year and group. Represents "how this word was used". |
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

---

*This document is a plain-language companion to the technical code. For questions about the methodology, refer to the notebook files `natureImportance_byDimension.ipynb` and `metaData.ipynb`.*

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

> ⚠️ Note: This project ignores `.env`, `env/`, `.ipynb_checkpoints/`, and `.DS_Store` files to keep the repository clean.

---

## 📁 Folder Descriptions

### **`/data/`**
Main folder containing pretrained embeddings and all data outputs.

#### **`/data/results/`**
Contains the data for different analyses:

##### **`/data/results/transcript`**
Analysis of the change in the language of the speeches for a country over the years

##### **`/data/results/metadata`**
Counting of various demographics of the speakers

##### **`/data/results/meta_transcript`**
After analysing the speeches country-wise and year-wise and counting the speakers across different classifications, we go a step further and analyse the behaviour within these demographics.

#### **`/data/raw/`**
This folder contains the **machine-translated English versions** of the respective country's ParlaMint parliamentary proceedings, organised by year.

> 💡 **Pickle** is used for Python-native objects like nested lists and arrays, allowing faster reloads and avoiding format errors.

---

## 📄 File Descriptions

### **`/nature_context.csv`**
Stores all context windows around the word **"nature"** across speeches and years.

**Each row includes:**
- Year of the speech  
- Filename of the source  
- Context snippet (±5 words around *"nature"*)  

**Purpose:**  
Provides raw data to compute the semantic embedding of *"nature"* over time and across parliaments.

---

### **`/process_parlamint.ipynb`**
Your **main Python notebook** that executes the entire analysis pipeline.

**What it does:**
- Loads and cleans speech data  
- Extracts context windows around target keywords  
- Computes ALC vectors using FastText  
- Calculates cosine similarity with reference terms  
- Saves all results and intermediate data for re-use  

**Purpose:**  
This notebook is your main control center. Reuse or adapt it for new countries, keywords, or embedding models.


---

### **`/requirements.txt`**
Lists all the Python packages required to run the project.

**Each line includes:**
- Package name  
- Pinned version (ensures reproducibility)

**Example:**
```text
pandas==2.2.1
numpy==1.26.4
```

**Purpose:**  
Allows anyone to recreate the exact Python environment by running:
```bash
pip install -r requirements.txt
```

> 🔄 Update this file by running `pip freeze > requirements.txt` after installing new packages.

---

### **`/data/raw/ParlaMint-{FR}-en.txt/`**
This folder contains the **machine-translated English versions** of the respective country's ParlaMint parliamentary proceedings, organised by year.

Each subfolder of a country (e.g., `2017/`) includes:
- **`.txt` files**: the actual **speech content** in English for each session.
- **`-meta.tsv` files**: associated **metadata** for each session, such as speaker ID, party affiliation, date, and session details.

**Example filenames:**
- `ParlaMint-FR-en_2017-07-03-O1001.txt` → English speech file  
- `ParlaMint-FR-en_2017-07-03-O1001-meta.tsv` → Metadata for the same session  

**Purpose:**  
These `.txt` files serve as the **raw input** for the pipeline — they are cleaned, parsed, and tokenized to extract context windows and compute yearly embeddings for key climate-related terms.

---

### **`/data/results/transcript/{country_name}/contexts_by_term_year.csv`**
Extracted ±5 word **context windows** around the keyword (e.g., *"nature"*) for each speech.

**Each row includes:**
- Year  
- Target work  
- Context snippet  

**Purpose:**  
Captures how the target word is used in discourse and sets up the data for ALC embedding.

---

### **`/data/results/{country_name}/alc_embeddings.pkl`**
Year-wise **ALC embeddings** for the target word.

**How it works:**
- Context windows are embedded using FastText  
- Each window's context vectors are averaged  
- All such vectors in a year are then averaged → 1 ALC vector per year  

**Format:** Python dictionary `{year: 300-dim NumPy vector}` saved as `.pkl`

**Purpose:**  
Represents how the **meaning** of a word shifts over the years in political discussions.

---

### **`/data/results/transcript/{country_name}/{year}_nature_vs_importance.csv`**
Records **cosine similarity** between yearly ALC embeddings and fixed reference terms (e.g., *"importance"*, *"emergency"*).

**Purpose:**  
Quantifies semantic closeness over time — helps understand shifts in **framing** of the word (e.g., does *"nature"* become more associated with *"emergency"*?).

---

### **`/data/embeddings/cc.en.300.vec`**
Pretrained **FastText English word embeddings**.

**Contents:**
- ~2 million words  
- 300-dimensional vectors per word  

**Format:** `.vec` file (text-based, ~6 GB)

**Purpose:**  
Used to embed context words numerically for computing ALC and cosine similarity.
