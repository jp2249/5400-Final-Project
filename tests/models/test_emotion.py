import pandas as pd
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../..", "Emotion-Model"))
from emotion_analysis import EmotionAnalyzer

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "quote": [
            "I am so remorseful for what I did",
            "I am happy to be going home",
            "This is a neutral statement"
        ],
        "id": [1, 2, 3]
    })

class TestEmotionAnalyzer:
    """Test suite for EmotionAnalyzer model"""

    def test_emotion_analyzer_initialization(self, sample_df):
        analyzer = EmotionAnalyzer(sample_df, "quote")
        assert analyzer.text_col == "quote"
        assert analyzer.df.shape == sample_df.shape

    def test_chunk_text(self, sample_df):
        analyzer = EmotionAnalyzer(sample_df, "quote", word_chunk_size=2)
        text = "one two three four five"
        chunks = analyzer.chunk_text(text)
        # expected: "one two", "three four", "five"
        assert len(chunks) == 3
        assert chunks[0] == "one two"

    # Note: We are mocking the pipeline to avoid downloading models during tests if possible, 
    # but for integration tests we might want the real model.
    # For now, let's skip the heavy model inference test unless we want to download the model.
    # We can check if get_top_emotion returns the right structure at least.

    def test_get_top_emotion_structure(self, sample_df):
        analyzer = EmotionAnalyzer(sample_df, "quote")
        # Using a very simple mock or just checking handling of empty text
        empty_res = analyzer.get_top_emotion(None)
        assert empty_res["emotion_label"] is None
        
        empty_res_2 = analyzer.get_top_emotion("")
        assert empty_res_2["emotion_label"] is None
