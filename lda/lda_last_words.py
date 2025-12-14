import logging
import os
import pandas as pd
from lda.lda_last_words_kgrid import LDALastWordsKGrid

def last_word_analysis(input_file="data/processed_data/last_words_data.csv", output_dir="data/processed_data",save=True):
    df = pd.read_csv(input_file)
    lda = LDALastWordsKGrid(output_dir=output_dir)
    lda.init_nlp()
    if "year" not in df.columns:
        df["year"] = df.apply(lda.year_from_row, axis=1)
    if "century_bucket" not in df.columns:
        df["century_bucket"] = df["year"].apply(lda.bucket_century)
    results = {}
    results["All"] = lda.model_subset(df, "All", save=save)
    logging.info("Last words topic for All: modeling complete")
    logging.info("*" * 80)
    for b in ("21st", "20th", "19th", "18th", "17th", "pre 17th"):
        tag = f"{b}_century"
        results[tag] = lda.model_subset(df[df["century_bucket"] == b], tag, save=save)
        logging.info(f"Last words topic for {b} century: modeling complete")
        logging.info("*" * 60)
    for col in ("is_religious", "is_criminal"):
        for val, sub in df.groupby(col):
            tag = col.replace("is_", "") if int(val) == 1 else f"not_{col.replace('is_', '')}"
            results[tag] = lda.model_subset(sub, tag, save=save)
            logging.info("*" * 80)

    return results, df, lda


def setup_logging(log_path="logs/lda_last_words.log"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)
    remove_log_libs = ["gensim", "spacy", "nltk", "urllib3", "matplotlib"]
    for lib in remove_log_libs:
        logging.getLogger(lib).setLevel(logging.ERROR)

if __name__ == "__main__":
    setup_logging()
    results, df, lda = last_word_analysis(input_file="data/processed_data/last_words_data.csv",output_dir="data/processed_data",save=True,
)
    print("LDA modeling complete.")