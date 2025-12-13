"""
LDA topic modeling on last_words_data.csv

Usage:
    python lda_last_words.py --csv path/to/lwd.csv
"""
import argparse
import pandas as pd
import numpy as np
import os
import sys

# Text + NLP
import nltk
from nltk.corpus import stopwords
import spacy

# Topic modeling
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from gensim.models.phrases import Phrases, Phraser


def preprocess_text(text, nlp, stop_words):
    text = text.lower()
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.is_punct or token.is_space or token.like_num:
            continue
        lemma = token.lemma_.strip()
        if len(lemma) < 2:
            continue
        if lemma in stop_words:
            continue
        tokens.append(lemma)
    return tokens


def fit_lda_and_coherence(corpus, dictionary, texts, num_topics, random_state=42):
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=random_state,
        passes=10,
        iterations=100,
        alpha="auto",
        eta="auto",
    )
    coherence_model = CoherenceModel(
        model=lda,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v",
    )
    coherence = coherence_model.get_coherence()
    return lda, coherence


def run_lda_analysis(df, nlp_model_name="en_core_web_sm"):
    """
    Runs the LDA analysis on the provided DataFrame.
    Returns the DataFrame with topic distributions.
    """
    if "quote" not in df.columns or "context" not in df.columns:
        raise ValueError("CSV must contain 'quote' and 'context' columns.")

    print("Downloading NLTK stopwords (if not already present)...")
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))

    df["quote"] = df["quote"].fillna("")
    df["context"] = df["context"].fillna("")
    df["document"] = df["quote"] + " " + df["context"]
    df = df[df["document"].str.strip() != ""]

    if df.empty:
        raise ValueError("All documents are empty after cleaning.")

    print(f"Loaded {len(df)} documents.")

    print(f"Loading spaCy model '{nlp_model_name}'...")
    try:
        nlp = spacy.load(nlp_model_name, disable=["parser", "ner"])
    except OSError:
        print(f"spaCy model '{nlp_model_name}' not found. Run:")
        print(f"    python -m spacy download {nlp_model_name}")
        raise

    print("Preprocessing documents (this may take a bit)...")
    tokenized_docs = [preprocess_text(t, nlp, stop_words) for t in df["document"].tolist()]

    print("Learning bigrams...")
    bigram_model = Phrases(tokenized_docs, min_count=5, threshold=10)
    bigram_phraser = Phraser(bigram_model)
    tokenized_docs_bigrams = [bigram_phraser[doc] for doc in tokenized_docs]

    print("Building dictionary and corpus...")
    dictionary = corpora.Dictionary(tokenized_docs_bigrams)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs_bigrams]

    if len(dictionary) == 0:
        raise ValueError("Dictionary is empty after filtering.")

    topic_range = [4, 6, 8]
    models = {}
    coherences = {}

    print("\nTraining LDA models and computing coherence:")
    for k in topic_range:
        print(f"  -> Fitting LDA with K = {k} ...")
        lda_k, coh_k = fit_lda_and_coherence(corpus, dictionary, tokenized_docs_bigrams, k)
        models[k] = lda_k
        coherences[k] = coh_k
        print(f"     K = {k}: coherence = {coh_k:.3f}")

    best_k = max(coherences.keys(), key=lambda k: coherences[k])
    lda_best = models[best_k]
    print(f"\nBest K by coherence is: K = {best_k} (coherence = {coherences[best_k]:.3f})")

    print(f"\nTop words for each topic (K = {best_k}):")
    for t in range(best_k):
        print(f"\nTopic #{t}")
        print(lda_best.print_topic(t, topn=10))

    def get_topic_distribution(lda_model, bow):
        return lda_model.get_document_topics(bow, minimum_probability=0.0)

    print("\nComputing topic distributions for each document...")
    all_topic_dists = [get_topic_distribution(lda_best, bow) for bow in corpus]

    topic_cols = [f"topic_{i}" for i in range(best_k)]
    topic_matrix = np.zeros((len(all_topic_dists), best_k))

    for i, dist in enumerate(all_topic_dists):
        for topic_id, prob in dist:
            topic_matrix[i, topic_id] = prob

    topic_df = pd.DataFrame(topic_matrix, columns=topic_cols)
    df_topics = pd.concat([df.reset_index(drop=True), topic_df], axis=1)

    label_cols = ["is_criminal", "is_religious", "is_expected"]
    for col in label_cols:
        if col in df_topics.columns:
            df_topics[col] = pd.to_numeric(df_topics[col], errors="coerce")

    for col in label_cols:
        if col in df_topics.columns:
            print(f"\nAverage topic proportions by {col}:")
            print(df_topics.groupby(col)[topic_cols].mean())

    return df_topics


def main():
    parser = argparse.ArgumentParser(description="LDA topic modeling")
    parser.add_argument("--csv", default="lwd.csv", help="Path to input CSV file")
    args = parser.parse_args()

    csv_path = args.csv
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found in current directory: {os.getcwd()}")
        sys.exit(1)

    print(f"Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path)

    try:
        df_topics = run_lda_analysis(df)
        out_path = os.path.join("data", "processed_data", "last_words_with_topics.csv")
        df_topics.to_csv(out_path, index=False)
        print(f"\nSaved dataframe with topic proportions to: {out_path}")
        print("Done.")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
