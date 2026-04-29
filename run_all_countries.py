#!/usr/bin/env python3
"""
run_all_countries.py
====================
Batch runner that re-runs the ALC analysis (with transformation matrix)
for ALL countries, covering both:
  1. natureImportance (country-level transcript analysis)
  2. natureImportance_byDimension (demographic-split analysis)

Usage:
    python run_all_countries.py

Prerequisites:
    - GloVe 840B.300d embeddings at data/embeddings/glove.840B.300d.txt
    - ALC transform matrix at data/alc_transform_glove840B.bin
      (pre-trained by NLPrinceton/ALaCarte for GloVe 840B.300d)
"""

import os, glob, re, string, sys, time
import pandas as pd
import numpy as np
from numpy.linalg import norm
from gensim.models import KeyedVectors

# ═══════════════════════════════════════════════════════════════════════
# 0. CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
BASE_DATA   = "/Users/harshit/Desktop/nature-in-language/data"
EMBED_PATH  = os.path.join(BASE_DATA, "embeddings", "glove.840B.300d.txt")
MATRIX_PATH = os.path.join(BASE_DATA, "alc_transform_glove840B.bin")
RAW_DIR     = os.path.join(BASE_DATA, "raw")

nature_terms     = ["nature","climate","environment","land","forest","forests",
                    "biodiversity","restoration","reforestation","ecology"]
importance_terms = ["important","importance","significant","meaningful"]
WINDOW = 10

# Generation ranges
GEN_RANGES = {
    'Silent_Generation': (1928, 1945),
    'Baby_Boomers':      (1946, 1964),
    'Generation_X':      (1965, 1980),
    'Millennials':       (1981, 1996),
    'Generation_Z':      (1997, 2012),
}
DIMENSIONS = ['gender', 'alignment', 'ideology', 'generation']


# ═══════════════════════════════════════════════════════════════════════
# 1. LOAD MODELS
# ═══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Loading GloVe 840B.300d embeddings…")
t0 = time.time()
ft_model = KeyedVectors.load_word2vec_format(EMBED_PATH, binary=False, no_header=True)
print(f"GloVe loaded: {len(ft_model.key_to_index):,} words, "
      f"{ft_model.vector_size} dims ({time.time()-t0:.1f}s)")

print("Loading ALC transform matrix (Princeton pre-trained for GloVe 840B)…")
import numpy as np
alc_matrix = np.fromfile(MATRIX_PATH, dtype=np.float32)
d = int(np.sqrt(alc_matrix.shape[0]))
alc_matrix = alc_matrix.reshape(d, d)
print(f"ALC transform matrix: {alc_matrix.shape}")

# Importance embeddings (static, shared across all runs)
imp_embeddings = {
    imp: ft_model[imp]
    for imp in importance_terms
    if imp in ft_model.key_to_index
}
print(f"Importance embeddings: {list(imp_embeddings.keys())}")


# ═══════════════════════════════════════════════════════════════════════
# 2. HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════
def clean_text(text):
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_contexts(tokens, years_sorted):
    """Extract context windows around nature terms."""
    contexts = {term: {yr: [] for yr in years_sorted} for term in nature_terms}
    # This is called per-row, so tokens is from a single doc
    return contexts


def compute_alc_embeddings(contexts_by_term_year, years_sorted):
    """Compute ALC embeddings WITH transformation matrix."""
    alc_emb = {term: {} for term in nature_terms}
    for term, year_dict in contexts_by_term_year.items():
        for yr, ctxs in year_dict.items():
            inst_vecs = []
            for ctx in ctxs:
                vs = [ft_model[w] for w in ctx if w in ft_model.key_to_index]
                if vs:
                    inst_vecs.append(np.sum(vs, axis=0))
            if inst_vecs:
                ctx_avg = np.sum(inst_vecs, axis=0) / len(inst_vecs)
                alc_emb[term][yr] = ctx_avg @ alc_matrix.T  # ALC transformation
    return alc_emb


def build_cosine_tables(alc_embeddings, years_sorted, out_dir, country):
    """Build and save 10×4 cosine similarity tables per year."""
    for yr in years_sorted:
        data = {}
        for term in nature_terms:
            if yr in alc_embeddings[term]:
                vec_n = alc_embeddings[term][yr]
                sims = {
                    imp: float(np.dot(vec_n, vec_i)/(norm(vec_n)*norm(vec_i)))
                    for imp, vec_i in imp_embeddings.items()
                }
                data[term] = sims
        df_year = pd.DataFrame.from_dict(data, orient='index').reindex(nature_terms)
        df_year.to_csv(os.path.join(out_dir, f"{yr}_nature_vs_importance.csv"), index=True)


def get_generation(birth_str):
    try:
        year = int(float(birth_str))
    except:
        return None
    for label, (s, e) in GEN_RANGES.items():
        if s <= year <= e:
            return label
    return None


def build_speaker_groups(speaker_meta, dimension):
    groups = {}
    for spk_id, row in speaker_meta.items():
        if dimension == 'gender':
            g = row.get('Speaker_gender', '').strip().upper()
            label = 'Male' if g == 'M' else ('Female' if g == 'F' else None)
        elif dimension == 'alignment':
            s = row.get('Party_status', '').strip()
            label = s if s in ('Coalition', 'Opposition') else None
        elif dimension == 'ideology':
            o = row.get('Party_orientation', '').strip().lower()
            if 'left' in o:              label = 'Left'
            elif 'right' in o:           label = 'Right'
            elif 'centre' in o or 'center' in o: label = 'Centre'
            else:                        label = None
        elif dimension == 'generation':
            label = get_generation(row.get('Speaker_birth', ''))
        else:
            label = None
        if label:
            groups.setdefault(label, set()).add(spk_id)
    return groups


# ═══════════════════════════════════════════════════════════════════════
# 3. PART A: COUNTRY-LEVEL ANALYSIS (natureImportance equivalent)
# ═══════════════════════════════════════════════════════════════════════
def run_transcript_analysis(country, corpus_dir):
    """Run country-level ALC analysis (no demographic splits)."""
    results_dir = os.path.join(BASE_DATA, "results", "transcript", country)
    os.makedirs(results_dir, exist_ok=True)

    # Load speeches
    speech_files = glob.glob(os.path.join(corpus_dir, "*/*.txt"))
    # Exclude meta files
    speech_files = [f for f in speech_files if not f.endswith("-meta.tsv")]
    texts, years = [], []
    for path in speech_files:
        years.append(os.path.basename(os.path.dirname(path)))
        with open(path, 'r', encoding='utf-8') as f:
            texts.append(f.read())

    if not texts:
        print(f"    [SKIP] No transcript files found")
        return

    df = pd.DataFrame({"year": years, "text": texts})
    df['text'] = df['text'].astype(str)
    df['clean_text'] = df['text'].apply(clean_text)

    years_sorted = sorted(df['year'].unique())

    # Extract context windows
    contexts_by_term_year = {
        term: {yr: [] for yr in years_sorted}
        for term in nature_terms
    }
    for _, row in df.iterrows():
        yr = row['year']
        tokens = row['clean_text'].split()
        for i, tok in enumerate(tokens):
            if tok in nature_terms:
                start = max(0, i - WINDOW)
                end   = min(len(tokens), i + WINDOW + 1)
                ctx   = tokens[start:i] + tokens[i+1:end]
                contexts_by_term_year[tok][yr].append(ctx)

    # Save contexts
    import pickle as _pkl
    with open(os.path.join(results_dir, "contexts_by_term_year.pkl"), "wb") as f:
        _pkl.dump(contexts_by_term_year, f)
    ctx_rows = [
        {"year": yr, "term": term, "context": " ".join(ctx)}
        for term, yd in contexts_by_term_year.items()
        for yr, ctxs in yd.items()
        for ctx in ctxs
    ]
    pd.DataFrame(ctx_rows).to_csv(
        os.path.join(results_dir, "contexts_by_term_year.csv"), index=False)

    # Compute ALC embeddings
    alc_emb = compute_alc_embeddings(contexts_by_term_year, years_sorted)
    with open(os.path.join(results_dir, "alc_embeddings.pkl"), "wb") as f:
        _pkl.dump(alc_emb, f)
    with open(os.path.join(results_dir, "imp_embeddings.pkl"), "wb") as f:
        _pkl.dump(imp_embeddings, f)

    # Build cosine tables
    build_cosine_tables(alc_emb, years_sorted, results_dir, country)
    print(f"    ✓ transcript: {len(years_sorted)} years, {len(df)} docs")


# ═══════════════════════════════════════════════════════════════════════
# 4. PART B: DIMENSION ANALYSIS (natureImportance_byDimension equivalent)
# ═══════════════════════════════════════════════════════════════════════
def run_dimension_analysis(country, corpus_dir):
    """Run demographic-split ALC analysis."""
    results_dir = os.path.join(BASE_DATA, "results", "meta_transcript", country)
    os.makedirs(results_dir, exist_ok=True)

    # Load metadata
    meta_frames = []
    for root, _, files in os.walk(corpus_dir):
        for fname in files:
            if fname.endswith('-meta.tsv'):
                try:
                    mdf = pd.read_csv(os.path.join(root, fname), sep='\t', index_col=False)
                    yr_match = re.search(r'_(\d{4})-\d{2}-\d{2}', fname)
                    mdf['_year'] = yr_match.group(1) if yr_match else ''
                    meta_frames.append(mdf)
                except Exception as e:
                    pass

    if not meta_frames:
        print(f"    [SKIP] No metadata files found")
        return

    meta_all = pd.concat(meta_frames, ignore_index=True)
    for col in ['Speaker_gender', 'Party_status', 'Party_orientation', 'Speaker_birth']:
        if col in meta_all.columns:
            meta_all[col] = meta_all[col].fillna('').astype(str)

    utterance_to_meta = (
        meta_all
        .set_index('ID')[['Speaker_ID', '_year', 'Speaker_gender',
                           'Party_status', 'Party_orientation', 'Speaker_birth']]
        .to_dict('index')
    )
    speaker_meta = (
        meta_all
        .drop_duplicates(subset='Speaker_ID')
        .set_index('Speaker_ID')[['Speaker_gender', 'Party_status', 'Party_orientation', 'Speaker_birth']]
        .to_dict('index')
    )

    # Load transcripts with speaker info
    speech_records = []
    txt_files = [p for p in glob.glob(os.path.join(corpus_dir, '*/*.txt'))
                 if not p.endswith('-meta.tsv')]

    for path in txt_files:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or '\t' not in line:
                    continue
                utterance_id, _, text = line.partition('\t')
                utterance_id = utterance_id.strip()
                if utterance_id not in utterance_to_meta:
                    continue
                meta_row = utterance_to_meta[utterance_id]
                speech_records.append({
                    'speaker_id': meta_row['Speaker_ID'],
                    'year': meta_row['_year'],
                    'clean_text': clean_text(text)
                })

    if not speech_records:
        print(f"    [SKIP] No matched utterances")
        return

    df_speeches = pd.DataFrame(speech_records)

    # Run for each dimension
    for dimension in DIMENSIONS:
        speaker_groups = build_speaker_groups(speaker_meta, dimension)
        for group_label, speaker_ids in speaker_groups.items():
            subset = df_speeches[df_speeches['speaker_id'].isin(speaker_ids)].copy()
            if subset.empty:
                continue

            out_dir = os.path.join(results_dir, dimension, group_label)
            os.makedirs(out_dir, exist_ok=True)

            years_sorted = sorted(subset['year'].unique())

            # Extract contexts
            contexts_by_term_year = {
                term: {yr: [] for yr in years_sorted}
                for term in nature_terms
            }
            for _, row in subset.iterrows():
                yr     = row['year']
                tokens = row['clean_text'].split()
                for i, tok in enumerate(tokens):
                    if tok in nature_terms:
                        start = max(0, i - WINDOW)
                        end   = min(len(tokens), i + WINDOW + 1)
                        ctx   = tokens[start:i] + tokens[i+1:end]
                        contexts_by_term_year[tok][yr].append(ctx)

            import pickle as _pkl
            with open(os.path.join(out_dir, 'contexts_by_term_year.pkl'), 'wb') as f:
                _pkl.dump(contexts_by_term_year, f)
            ctx_rows = [
                {'year': yr, 'term': term, 'context': ' '.join(ctx)}
                for term, yd in contexts_by_term_year.items()
                for yr, ctxs in yd.items()
                for ctx in ctxs
            ]
            pd.DataFrame(ctx_rows).to_csv(
                os.path.join(out_dir, 'contexts_by_term_year.csv'), index=False)

            # Compute ALC embeddings with transformation
            alc_emb = compute_alc_embeddings(contexts_by_term_year, years_sorted)
            with open(os.path.join(out_dir, 'alc_embeddings.pkl'), 'wb') as f:
                _pkl.dump(alc_emb, f)
            with open(os.path.join(out_dir, 'imp_embeddings.pkl'), 'wb') as f:
                _pkl.dump(imp_embeddings, f)

            # Build cosine tables
            build_cosine_tables(alc_emb, years_sorted, out_dir, country)

            print(f"      ✓ {dimension}/{group_label}: "
                  f"{len(years_sorted)} years, {len(subset):,} utterances")


# ═══════════════════════════════════════════════════════════════════════
# 5. MAIN: RUN ALL COUNTRIES
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    country_dirs = sorted(glob.glob(os.path.join(RAW_DIR, "ParlaMint-*-en.txt")))
    countries = [
        os.path.basename(d).replace("ParlaMint-", "").replace("-en.txt", "")
        for d in country_dirs
    ]

    print(f"\n{'=' * 70}")
    print(f"RUNNING ALC ANALYSIS FOR {len(countries)} COUNTRIES")
    print(f"Countries: {', '.join(countries)}")
    print(f"{'=' * 70}\n")

    for idx, (country, corpus_dir) in enumerate(zip(countries, country_dirs), 1):
        print(f"\n[{idx}/{len(countries)}] === {country} ===")
        t_start = time.time()

        try:
            run_transcript_analysis(country, corpus_dir)
        except Exception as e:
            print(f"    ✗ transcript FAILED: {e}")

        try:
            run_dimension_analysis(country, corpus_dir)
        except Exception as e:
            print(f"    ✗ dimension FAILED: {e}")

        elapsed = time.time() - t_start
        print(f"    Done in {elapsed:.1f}s")

    print(f"\n{'=' * 70}")
    print("ALL COUNTRIES COMPLETE!")
    print(f"{'=' * 70}")
