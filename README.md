# 5400-Final-Project

Team Members:
- Jeffrey Pinarchick
- Tyler McCormick
- Younghoon Kim
- Sam Gold
- Nikhil Poluri

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

### Prerequisites
- Python ^3.10
- Poetry

### Setup

1.  **Install Dependencies**:
    ```bash
    poetry install
    ```

2.  **Activate Virtual Environment**:
    ```bash
    poetry shell
    ```

## Running Tests

To run the test suite, execute the following command from the root directory:

```bash
pytest
```

## Tree of the project

```text
.
├── data
│   ├── merged_data
│   ├── processed_data
│   │   ├── ~$label_last_words_18th_century.xlsx
│   │   ├── label_last_words_18th_century.xlsx
│   │   ├── label_last_words_19th_century.xlsx
│   │   ├── label_last_words_20th_century.xlsx
│   │   ├── label_last_words_21st_century.xlsx
│   │   ├── label_last_words_ironic.xlsx
│   │   ├── label_last_words_notable.xlsx
│   │   ├── label_last_words_pre5_to_17_century.xlsx
│   │   ├── last_words_data_with_emotion (1).csv
│   │   ├── last_words_data.csv
│   │   ├── last_words_of_the_executed_labeled.csv
│   │   ├── relig_label_last_words_19th_century.xlsx
│   │   ├── relig_label_last_words_ironic.xlsx
│   │   ├── relig_label_last_words_notable.xlsx
│   │   └── relig_label_last_words_pre5_to_17_century.xlsx
│   └── raw_data
│       ├── last_words_18th_century.csv
│       ├── last_words_19th_century.csv
│       ├── last_words_20th_century.csv
│       ├── last_words_21st_century.csv
│       ├── last_words_ironic.csv
│       ├── last_words_notable.csv
│       ├── last_words_of_the_executed.csv
│       ├── last_words_pre5_to_17_century.csv
│       ├── processed
│       │   ├── test.csv
│       │   └── train.csv
│       ├── raw
│       │   └── last_words_data.csv
│       ├── texas_execution_dates_with_statements.csv
│       ├── texas_execution_dates.csv
│       ├── texas_labeled_final_with_NA.csv
│       └── texas_last_statements.csv
├── pyproject.toml
├── pytest.ini
├── README.md
├── src
│   ├── data_process
│   │   ├── texas_collect_last_word.py
│   │   ├── texas_label_clean.py
│   │   ├── text_cleaning.ipynb
│   │   └── wikipedia_scraping.ipynb
│   └── models
│       ├── criminal_classifier
│       │   ├── main.py
│       │   ├── model
│       │   │   ├── __init__.py
│       │   │   ├── __pycache__
│       │   │   │   ├── __init__.cpython-312.pyc
│       │   │   │   └── class_model.cpython-312.pyc
│       │   │   ├── class_model.py
│       │   │   ├── eval
│       │   │   │   ├── __init__.py
│       │   │   │   ├── __pycache__
│       │   │   │   │   ├── __init__.cpython-312.pyc
│       │   │   │   │   └── eval.cpython-312.pyc
│       │   │   │   └── eval.py
│       │   │   └── utils
│       │   │       ├── __init__.py
│       │   │       ├── __pycache__
│       │   │       │   ├── __init__.cpython-312.pyc
│       │   │       │   ├── load_data.cpython-312.pyc
│       │   │       │   └── visualize.cpython-312.pyc
│       │   │       ├── load_data.py
│       │   │       └── visualize.py
│       │   ├── run_code.txt
│       │   └── setup.py
│       ├── emotion_model
│       │   ├── emotion_analysis.py
│       │   ├── emotions_anly_oop.ipynb
│       │   └── sentiment_emotion_anly_raw.ipynb
│       └── lda
│           ├── lda_last_words_kgrid.py
│           └── lda_last_words.py
└── tests
    └── models
        ├── test_criminal.py
        └── test_emotion.py
```

## Project Status

## Project Structure & Setup
- [x] Initialize with `pyproject.toml` (not setup.py)
- [ ] Create `environment.yml` file with all dependencies
- [x] Set up proper directory structure (src/, tests/, data/, docs/, etc.)

## README Documentation
- [x] List full names of all team members
- [ ] Project aim and research question
- [x] Installation instructions
- [ ] Usage examples with code snippets
- [ ] Architecture diagram (created with draw.io or similar)
- [ ] Data download/access instructions
- [ ] Dependencies and environment setup guide

## Data Management
- [ ] Set up data storage (GitHub LFS or Google drive link)
- [x] Create data download script if possible
- [x] Implement data preprocessing pipeline
- [x] Store preprocessed data in folder
- [ ] Document data sources and preprocessing steps
- [ ] NO data pushed to GitHub

## Code Implementation
- [ ] Design OOP (classes for model, preprocessing, evaluation, etc.)
- [ ] Implement sentiment analysis model(s)
- [ ] Add comments to all functions, methods, and classes
- [ ] Implement logging at key points (data loading, training, evaluation)
- [ ] Follow PEP 8 standards using pylint (clean structure)

## Testing Suite
- [x] Create `tests/` directory with pytest structure
- [x] Set separate test dataset
- [x] Write tests for model components
- [ ] All tests pass

## Sentiment Analysis Specific
- [ ] Data loading and text preprocessing (tokenization + cleaning)
- [ ] Embeddings
- [ ] Model implementation (baseline + advanced models)
- [ ] Training pipeline
- [ ] Evaluation metrics (accuracy, F1, precision, recall, confusion matrix)
- [ ] Results visualization and analysis
