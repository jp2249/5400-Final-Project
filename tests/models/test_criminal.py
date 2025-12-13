# need to test the other helper functions as well?
from criminal_classifier.model.class_model import CriminalIdentifier
from criminal_classifier.model.eval.eval import evaluate
from criminal_classifier.model.utils.load_data import csv_df, train_test_df
import numpy as np
import pandas as pd
import pytest

# run command: `pytest tests/ -v` from the root directory
# to test code using pytest

# making sample test data
@pytest.fixture
def train_data():
    return pd.DataFrame({
        'quote': ['today is a good day',
                  'i did not commit the crime',
                  'what is for breakfast',
                  'i am innocent, i promise',
                  'tell my family i love them'],
        'is_criminal': [0, 1, 0, 1, 1]
    })

@pytest.fixture 
def test_data():
    """Test data fixture"""
    return pd.DataFrame({
        'quote': ['i would like some water',
                  'i am going to a better place',
                  'hello dear'],
        'is_criminal': [0, 1, 0]
    })

class TestCriminalIdentifier:
    """Test suite for CriminalIdentifier model"""
    
    # testing that the training function works
    def test_training(self, train_data):
        model = CriminalIdentifier()
        model.train_df(train_data)

        # two classes for criminal or not
        assert len(model.classes_) == 2

    # testing tha the predictions work
    def test_predictions(self, train_data, test_data):
        model = CriminalIdentifier()
        model.train_df(train_data)
        probs = model.pred_probabilities(test_data)

        # checking probablities are outputting correctly
        # for the right number of quotes
        # sum of pair of probabilities should equal to 1
        assert probs.shape == (len(test_data), 2)
        assert np.allclose(probs.sum(axis = 1), 1.0)
