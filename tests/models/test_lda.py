import pytest
import pandas as pd
from unittest.mock import patch
from lda.lda_last_words import last_word_analysis


@pytest.fixture
def df_ok():
    return pd.DataFrame({
        "quote": ["a", "b", "c"],
        "context": ["x", "y", "z"],
        "is_criminal": [1, 0, 1],
        "is_religious": [0, 1, 0],
        "is_expected": [1, 0, 1],
        "year": [2001, 1999, 1805],          # <- prevents df.apply(lda.year_from_row, ...)
        "century_bucket": ["21st", "20th", "19th"],  # <- prevents df["year"].apply(lda.bucket_century)
    })

@pytest.fixture
def df_empty():
    return pd.DataFrame(columns=[
        "quote", "context", "is_criminal", "is_religious",
        "year", "century_bucket"
    ])

class TestLastWordAnalysis:
    @patch("lda.lda_last_words.pd.read_csv")
    @patch("lda.lda_last_words.LDALastWordsKGrid")
    def test_last_word_analysis_happy_path(self, MockLDA, mock_read_csv, df_ok):
        mock_read_csv.return_value = df_ok

        lda = MockLDA.return_value
        lda.model_subset.return_value = df_ok.assign(topic_0=0.5)  # pretend modeling succeeded

        results, df_out, lda_out = last_word_analysis("data/processed/last_words_data.csv", save=False)

        assert "All" in results
        assert results["All"] is not None
        assert "quote" in df_out.columns
        assert "context" in df_out.columns
        lda.init_nlp.assert_called_once()
        assert lda_out is lda

    @patch("lda.lda_last_words.pd.read_csv")
    @patch("lda.lda_last_words.LDALastWordsKGrid")
    def test_last_word_analysis_empty_file(self, MockLDA, mock_read_csv, df_empty):
        mock_read_csv.return_value = df_empty

        lda = MockLDA.return_value
        lda.model_subset.return_value = None  # modeling returns None on empty subsets

        results, df_out, _ = last_word_analysis("fake.csv", save=False)

        assert "All" in results
        assert results["All"] is None
        assert df_out.empty
