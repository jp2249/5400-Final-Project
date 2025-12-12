# 5400-Final-Project

Team Members:
- Jeffrey Pinarchick
- Tyler McCormick
- Younghoon Kim
- Sam Gold
- Nikhil Poluri

## Setup and Installation

This project uses **Conda** for environment management and **pip** for package installation. The code is organized into the `src/` directory.

### Prerequisites
- **Anaconda or Miniconda**: [Installation Guide](https://docs.anaconda.com/free/miniconda/)

### Installation Steps

1.  **Create Conda Environment**  
    Create the environment from the `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    ```

2.  **Activate Environment**  
    ```bash
    conda activate 5400-final-project
    ```

3.  **Install Packages in Editable Mode**  
    Install the project in editable mode. This allows you to edit the code and see changes immediately without reinstalling.
    ```bash
    pip install -e .
    ```

4.  **Download Language Models**  
    This project requires the English spaCy model.
    ```bash
    python -m spacy download en_core_web_sm
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
│   ├── processed_data
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
│   │   ├── relig_label_last_words_pre5_to_17_century.xlsx
│   │   └── train.csv
│   └── raw_data
│       ├── last_words_18th_century.csv
│       ├── last_words_19th_century.csv
│       ├── last_words_20th_century.csv
│       ├── last_words_21st_century.csv
│       ├── last_words_data.csv
│       ├── last_words_ironic.csv
│       ├── last_words_notable.csv
│       ├── last_words_of_the_executed.csv
│       ├── last_words_pre5_to_17_century.csv
│       ├── texas_execution_dates_with_statements.csv
│       ├── texas_execution_dates.csv
│       ├── texas_labeled_final_with_NA.csv
│       └── texas_last_statements.csv
├── environment.yml
├── pyproject.toml
├── pytest.ini
├── README.md
├── src
│   ├── data_process
│   │   ├── __init__.py
│   │   ├── texas_collect_last_word.py
│   │   ├── texas_label_clean.py
│   │   ├── text_cleaning.ipynb
│   │   └── wikipedia_scraping.ipynb
│   └── models
│       ├── __init__.py
│       ├── criminal_classifier
│       ├── emotion_model
│       └── lda
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
- [ ] Implement data preprocessing pipeline
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
- [ ] Write tests for model components
- [ ] All tests pass

## Sentiment Analysis Specific
- [ ] Data loading and text preprocessing (tokenization + cleaning)
- [ ] Embeddings
- [ ] Model implementation (baseline + advanced models)
- [ ] Training pipeline
- [ ] Evaluation metrics (accuracy, F1, precision, recall, confusion matrix)
- [ ] Results visualization and analysis
