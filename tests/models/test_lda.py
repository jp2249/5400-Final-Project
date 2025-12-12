
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from models.lda.lda_last_words import run_lda_analysis

@pytest.fixture
def mock_df():
    return pd.DataFrame({
        "quote": ["hello world", "goodbye world", "justice served", "another quote", "last words"],
        "context": ["context one", "context two", "context three", "context four", "context five"],
        "is_criminal": [1, 0, 1, 0, 1],
        "is_religious": [0, 1, 0, 0, 1],
        "is_expected": [1, 1, 0, 0, 1]
    })

@patch("models.lda.lda_last_words.spacy")
@patch("models.lda.lda_last_words.nltk")
@patch("models.lda.lda_last_words.LdaModel")
@patch("models.lda.lda_last_words.CoherenceModel")
@patch("models.lda.lda_last_words.corpora.Dictionary")
@patch("models.lda.lda_last_words.Phrases")
@patch("models.lda.lda_last_words.Phraser")
def test_run_lda_analysis(mock_phraser, mock_phrases, mock_dict, mock_coh, mock_lda, mock_nltk, mock_spacy, mock_df):
    # Setup mocks
    # Mock spacy
    mock_nlp = MagicMock()
    mock_spacy.load.return_value = mock_nlp
    
    # Mock tokens
    doc_mock = MagicMock()
    token_mock = MagicMock()
    token_mock.lemma_ = "word"
    token_mock.is_punct = False
    token_mock.is_space = False
    token_mock.like_num = False
    doc_mock.__iter__.return_value = [token_mock]
    mock_nlp.return_value = doc_mock
    
    # Mock Dictionary
    mock_dictionary_instance = MagicMock()
    mock_dictionary_instance.__len__.return_value = 10
    mock_dict.return_value = mock_dictionary_instance
    
    # Mock LDA Model
    mock_lda_instance = MagicMock()
    # Mock get_document_topics to return list of (topic_id, prob)
    # Assume 4 topics
    mock_lda_instance.get_document_topics.return_value = [(0, 0.25), (1, 0.25), (2, 0.25), (3, 0.25)]
    mock_lda_instance.print_topic.return_value = "0.1*word + 0.1*other"
    mock_lda.return_value = mock_lda_instance
    
    # Mock Coherence Model
    mock_coh_instance = MagicMock()
    mock_coh_instance.get_coherence.return_value = 0.5
    mock_coh.return_value = mock_coh_instance
    
    # Run functionality
    result_df = run_lda_analysis(mock_df, nlp_model_name="en_core_web_sm")
    
    # Assertions
    assert not result_df.empty
    # We expect topics from K=4 (since it's in the list and we mocked constant coherence, it might pick any, but max of equal is first? or last?)
    # Actually max used key with max val.
    # Logic in code: topic_range = [4, 6, 8]
    # We return constant coherence 0.5. 'max' will just pick one.
    
    # Check if any topic columns exist
    topic_cols = [c for c in result_df.columns if c.startswith("topic_")]
    assert len(topic_cols) > 0
    
    # Check if original columns are preserved
    assert "quote" in result_df.columns
    assert "context" in result_df.columns

    # Check if topic values are populated
    assert not result_df[topic_cols].isnull().all().all()

def test_run_lda_analysis_validation():
    # Test empty dataframe
    empty_df = pd.DataFrame(columns=["quote", "context"])
    with pytest.raises(ValueError, match="All documents are empty"):
        # We need to create a slightly non-empty one that becomes empty or check logic
        # Actually logic says: if df.empty raise.
        # So pass empty df
        # But run_lda checks columns first.
         run_lda_analysis(empty_df)

    # Test missing columns
    bad_df = pd.DataFrame({"foo": [1, 2]})
    with pytest.raises(ValueError, match="must contain 'quote' and 'context'"):
        run_lda_analysis(bad_df)
