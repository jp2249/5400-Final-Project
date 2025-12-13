#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LDA topic modeling for 'last words' with automatic K-grid tuning per subset.
"""

import argparse
import os
import re
import sys
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import spacy
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from gensim.models.phrases import Phrases, Phraser

# ----------------------------
# Year extraction
# ----------------------------
def extract_year(raw):
    if raw is None:
        return np.nan
    s = str(raw).lower().strip()
    if s == "":
        return np.nan
    s = re.sub(r"\[.*?\]|\(.*?\)", "", s)
    s = (s.replace("o.s.", "").replace("c.", "").replace("ca.", "")
           .replace("circa", "").replace("?", "").strip())
    century = re.search(r"(\d+)(st|nd|rd|th)\s+century", s)
    if century:
        c = int(century.group(1))
        y = (c - 1) * 100 + 50
        return -y if "bc" in s else y
    if "bc" in s:
        m = re.search(r"\d{1,4}", s)
        return -int(m.group()) if m else np.nan
    s = s.replace("ad", "").strip()
    m4 = re.findall(r"\b\d{4}\b", s)
    if m4:
        return int(m4[-1])
    m2 = re.search(r"\b(\d{2})\b$", s)
    if m2:
        y = int(m2.group(1))
        return 2000 + y if y <= 25 else 1900 + y
    return np.nan

def extract_year_row(row):
    for col in ["date", "title", "context"]:
        if col in row and pd.notna(row[col]):
            y = extract_year(row[col])
            if not pd.isna(y):
                return y
    return np.nan

def bucket_century(y):
    if pd.isna(y):
        return "unknown"
    y = int(y)
    if y <= 1600: return "pre 17th"
    if 1601 <= y <= 1700: return "17th"
    if 1701 <= y <= 1800: return "18th"
    if 1801 <= y <= 1900: return "19th"
    if 1901 <= y <= 2000: return "20th"
    if 2001 <= y <= 2100: return "21st"
    return "unknown"

# ----------------------------
# K-grid selection
# ----------------------------
def pick_k_grid(n_docs: int, vocab_size: int):
    if n_docs < 80 or vocab_size < 600:
        grid = [2,3,4,5]
    elif n_docs < 200 or vocab_size < 1500:
        grid = [3,4,5,6,7]
    elif n_docs < 800 or vocab_size < 4000:
        grid = [4,5,6,8,10]
    else:
        grid = [6,8,10,12,15]
    k_cap = max(5, vocab_size // 30)
    grid = [k for k in grid if k <= k_cap and vocab_size >= k * 10]
    if not grid:
        grid = [2,3] if vocab_size >= 30 else [2]
    return sorted(set(grid))

def fit_lda_and_coherence_multi(corpus, dictionary, texts_tokenized, num_topics, seeds=(41,42,43)):
    scores, models = [], []
    for rs in seeds:
        lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics,
                       random_state=rs, passes=10, iterations=100, alpha="auto", eta="auto")
        cm = CoherenceModel(model=lda, texts=texts_tokenized, dictionary=dictionary, coherence="c_v")
        scores.append(cm.get_coherence())
        models.append(lda)
    best_i = int(np.argmax(scores))
    return models[best_i], float(np.mean(scores))

# ----------------------------
# LDA pipeline
# ----------------------------
def preprocess_and_corpus(df, nlp, stop_words, no_below=5, no_above=0.5):
    texts = (df["quote"].fillna("") + " " + df["context"].fillna("")).str.strip()
    texts = texts[texts != ""]
    if texts.empty: return None, None, None, None
    def tok(text):
        doc = nlp(text.lower())
        out = []
        for t in doc:
            if t.is_space or t.is_punct or t.like_num: continue
            lemma = t.lemma_.strip()
            if len(lemma) < 2 or lemma in stop_words: continue
            out.append(lemma)
        return out
    tokenized = [tok(t) for t in texts.tolist()]
    phraser = Phraser(Phrases(tokenized, min_count=5, threshold=10))
    tokenized_bi = [phraser[doc] for doc in tokenized]
    dictionary = corpora.Dictionary(tokenized_bi)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    if len(dictionary) == 0: return texts.index, tokenized_bi, None, None
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_bi]
    return texts.index, tokenized_bi, dictionary, corpus

def run_lda_block(df_block, tag, nlp, stop_words, outdir=None):
    """
    Runs LDA on a subset of data (df_block).
    Returns the DataFrame with topic distributions or None if skipped.
    If outdir is provided, saves the result to a CSV.
    """
    if df_block is None or df_block.empty:
        print(f"[{tag}] SKIP: empty subset."); return None
    
    idx, tokenized, dictionary, corpus = preprocess_and_corpus(df_block, nlp, stop_words)
    if dictionary is None or corpus is None or len(dictionary)==0:
        print(f"[{tag}] SKIP: empty dictionary."); return None
    
    n_docs, vocab_size = len(corpus), len(dictionary)
    topic_range = pick_k_grid(n_docs, vocab_size)
    best_k, best_coh, best_model = None, -1, None
    print(f"\n[{tag}] docs={n_docs}, vocab={vocab_size}, Ks={topic_range}")
    
    for k in topic_range:
        lda_k, coh_k = fit_lda_and_coherence_multi(corpus, dictionary, tokenized, k)
        print(f"  -> K={k:>2} mean_c_v={coh_k:.3f}")
        if coh_k > best_coh: best_k,best_coh,best_model = k,coh_k,lda_k
        
    if best_model is None: return None
    
    print(f"[{tag}] Best K={best_k} coherence={best_coh:.3f}")
    topic_cols=[f"topic_{i}" for i in range(best_k)]
    mat=np.zeros((len(corpus),best_k))
    for i,bow in enumerate(corpus):
        for t,p in best_model.get_document_topics(bow,minimum_probability=0.0):
            mat[i,t]=p
            
    topic_df=pd.DataFrame(mat,columns=topic_cols,index=idx)
    out=pd.concat([df_block.reset_index(drop=True),topic_df.reset_index(drop=True)],axis=1)
    
    for col in ["is_criminal","is_religious","is_expected"]:
        if col in out.columns:
            out[col]=pd.to_numeric(out[col],errors="coerce")
            print(f"\n[{tag}] Avg topic props by {col}:")
            print(out.groupby(col)[topic_cols].mean())
            
    if outdir:
        fname=os.path.join(outdir,f"last_words_with_topics__{tag}.csv")
        out.to_csv(fname,index=False)
        print(f"[{tag}] Saved {fname}")
        
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv",type=str,default="lwd.csv")
    args=ap.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found.")
        sys.exit(1)
        
    df=pd.read_csv(args.csv)
    for c in ["quote","context"]:
        if c not in df.columns:
            print(f"Missing column: {c}"); sys.exit(1)
            
    if "year" not in df.columns:
        df["year"]=df.apply(extract_year_row,axis=1)
    df["century_bucket"]=df["year"].apply(bucket_century)
    
    # Load resources once
    print("Downloading NLTK stopwords...")
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))
    
    print("Loading spaCy model...")
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser","ner"])
    except OSError:
        print("Run: python -m spacy download en_core_web_sm"); sys.exit(1)

    # Run blocks
    run_lda_block(df,"ALL", nlp, stop_words, outdir=".")
    
    for b in ["21st","20th","19th","18th","17th","pre 17th"]:
        run_lda_block(df[df["century_bucket"]==b],f"century={b.replace(' ','_')}", nlp, stop_words, outdir=".")
        
    for label in ["is_religious","is_criminal"]:
        if label in df.columns:
            for val,sub in df.groupby(label):
                tag=f"{label}={val if not pd.isna(val) else 'NA'}"
                run_lda_block(sub,tag, nlp, stop_words, outdir=".")
                
    print("Done.")

if __name__=="__main__":
    main()
