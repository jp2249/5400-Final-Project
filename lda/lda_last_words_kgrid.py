import os
import re
import sys
import numpy as np
import pandas as pd
import nltk
import spacy
import logging
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from gensim.models.phrases import Phrases, Phraser
from nltk.corpus import stopwords

class LDALastWordsKGrid:
    def __init__(
        self, seeds=(41, 42, 43), passes=10, iterations=100, min_docs_per_word=5, max_doc_fraction=0.5,phrase_min_count=5,
    phrase_threshold=10, output_dir="data/processed_data"):
        self.seeds = seeds
        self.passes = passes
        self.iterations = iterations
        self.min_docs_per_word = min_docs_per_word
        self.max_doc_fraction = max_doc_fraction
        self.phrase_min_count = phrase_min_count
        self.phrase_threshold = phrase_threshold
        self.stop_words = None
        self.nlp = None
        self.output_dir=output_dir

    def init_nlp(self):
        nltk.download("stopwords", quiet=True)
        self.stop_words = set(stopwords.words("english"))
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
 
    def extract_year(self, raw):
        if not raw:
            return np.nan
        s = re.sub(r"\[.*?\]|\(.*?\)", "", str(raw).lower()).strip()
        for x in ("o.s.", "c.", "ca.", "circa", "?"):
            s = s.replace(x, "")
        s = s.strip()
        m = re.search(r"(\d+)(st|nd|rd|th)\s+century", s)
        if m:
            y = (int(m.group(1)) - 1) * 100 + 50
            return -y if "bc" in s else y
        if "bc" in s:
            m = re.search(r"\d{1,4}", s)
            return -int(m.group()) if m else np.nan
        s = s.replace("ad", "").strip()
        m = re.findall(r"\b\d{4}\b", s)
        if m:
            return int(m[-1])
        m = re.search(r"\b(\d{2})\b$", s)
        if m:
            y = int(m.group(1))
            return 2000 + y if y <= 25 else 1900 + y
        return np.nan

    def year_from_row(self, row):
        for col in ("date", "title", "context"):
            y = self.extract_year(row.get(col))
            if not pd.isna(y):
                return y
        return np.nan

    def bucket_century(self, y):
        if pd.isna(y):
            return "century data missing"
        y = int(y)
        if y <= 1600:
            return "pre 17th"
        if 1601 <= y <= 1700:
            return "17th"
        if 1701 <= y <= 1800:
            return "18th"
        if 1801 <= y <= 1900:
            return "19th"
        if 1901 <= y <= 2000:
            return "20th"
        if 2001 <= y <= 2100:
            return "21st"
        return "century data missing"

    def k_candidates(self, n_docs, vocab_size):
        if n_docs < 80 or vocab_size < 600:
            grid = [2, 3, 4, 5]
        elif n_docs < 200 or vocab_size < 1500:
            grid = [3, 4, 5, 6, 7]
        elif n_docs < 800 or vocab_size < 4000:
            grid = [4, 5, 6, 8, 10]
        else:
            grid = [6, 8, 10, 12, 15]
        k_cap = max(5, vocab_size / 30)
        grid = [k for k in grid if k <= k_cap and vocab_size >= k * 10]
        if not grid:
            grid = [2, 3] if vocab_size >= 30 else [2]
        return sorted(set(grid))

    def tokenize_text(self, text):
        tokens = []
        for t in self.nlp((text or "").lower()):
            if t.is_space or t.is_punct or t.like_num:
                continue
            w = (t.lemma_ or "").strip()
            if len(w) < 2 or w in self.stop_words:
                continue
            tokens.append(w)
        return tokens

    def make_corpus(self, df):
        docs = (df["quote"].fillna("") + " " + df["context"].fillna("")).str.strip()
        docs = docs[docs != ""]
        if docs.empty:
            return None
        tks = [self.tokenize_text(d) for d in docs.tolist()]
        phrases = Phraser(Phrases(tks, min_count=self.phrase_min_count, threshold=self.phrase_threshold))
        tks = [phrases[t] for t in tks]
        dict = corpora.Dictionary(tks)
        dict.filter_extremes(no_below=self.min_docs_per_word,no_above=self.max_doc_fraction)
        if not dict:
            return {"idx": docs.index, "tokenized": tks, "dictionary": None, "corpus": None}
        crps = [dict.doc2bow(t) for t in tks]
        return {"idx": docs.index, "tokenized": tks, "dictionary": dict, "corpus": crps}

    def train_lda(self, corpus, dictionary, tokenized, k):
        scores, models = [], []
        for rs in self.seeds:
            lda = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=k,
                random_state=rs,
                passes=self.passes,
                iterations=self.iterations,
                alpha="auto",
                eta="auto",
            )
            cm = CoherenceModel(model=lda, texts=tokenized, dictionary=dictionary, coherence="c_v")
            scores.append(cm.get_coherence())
            models.append(lda)
        return models[int(np.argmax(scores))], float(np.mean(scores))
    
    def show_topics(self, model, k, topn=10):
        logging.info(f"Top words for each topic (K = {k}):")
        for i in range(k):
            logging.info(f"Topic #{i}:")
            logging.info("-" * 10)
            logging.info(model.print_topic(i, topn=topn))

    def select_k(self, corpus, dictionary, tokenized, tag=""):
        ks = self.k_candidates(len(corpus), len(dictionary))
        best = (None, -1.0, None) 
        logging.info(f"Training LDA models for {tag} : docs={len(corpus)}, vocab={len(dictionary)}, Ks={ks}")
        for k in ks:
            model, coh = self.train_lda(corpus, dictionary, tokenized, k)
            logging.info(f"Selecting LDA with K={k:>2}: Coherence={coh:.3f}")
            if coh > best[1]:
                best = (k, coh, model)
        k, coh, model = best
        if model is None:
            return None
        logging.info(f"Best K by Coherence is: K={k}, Coherence={coh:.3f}")
        return k, model

    def doc_topic_matrix(self, model, corpus, k):
        mat = np.zeros((len(corpus), k))
        for i, bow in enumerate(corpus):
            for tid, p in model.get_document_topics(bow, minimum_probability=0.0):
                mat[i, tid] = p
        return mat

    def model_subset(self, df, tag, save=True):
        if df is None or df.empty:
            logging.warning(f"[{tag}] Continue Data Frame is Empty")
            return None
        built = self.make_corpus(df)
        if built is None:
            logging.warning(f"[{tag}] Continue no corpus")
            return None
        if not built["dictionary"] or not built["corpus"]:
            logging.warning(f"[{tag}] Continue dictionary is empty")
            return None
        idx = built["idx"]
        tks = built["tokenized"]
        dict = built["dictionary"]
        crps = built["corpus"]
        tuned = self.select_k(crps, dict, tks, tag=tag)
        if not tuned:
            return None
        k, model = tuned
        self.show_topics(model, k) 
        topic_cols = [f"topic_{i}" for i in range(k)]
        topic_df = pd.DataFrame(self.doc_topic_matrix(model, crps, k), columns=topic_cols, index=idx)
        out = pd.concat([df.reset_index(drop=True), topic_df.reset_index(drop=True)], axis=1)
        for col in ("is_criminal", "is_religious", "is_expected"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
                out = out[out[col].isin([0, 1])]
                logging.info(f"[{tag}] Avg topic proportions by {col}")
                logging.info("-" * 40)
                avg_topic_dist = out.groupby(col)[topic_cols].mean().round(3)
                dist_table = avg_topic_dist.to_string()
                dist_table = "\n".join("    " + line for line in dist_table.splitlines())
                logging.info(f"[{tag}] Average topic proportions by {col}:\n{dist_table}")

                logging.info(out.groupby(col)[topic_cols].mean().round(3))
        if save and self.output_dir:
            path = os.path.join(self.output_dir, f"last_words_with_topics_{tag}.csv")
            out.to_csv(path, index=False)
        return out