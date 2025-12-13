## Check List for Cleaning Repo:

- [ ] Change all file into .csv
- [ ] Cleaned Single file for Wiki
- [ ] Cleaned Single file for Texas
- [ ] Single file for merging Wiki and Texas
  
- [ ] Remove unused *.csv and *.xlsx
- [ ] Decide Google drive or Git LFS
  
- [ ] Check each model for any modification
- [ ] Check each model still works or not


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

## Running the Criminal Identifier model

To run the entire workflow on the original data, execute the following command from the root directory:

```bash
python src/models/criminal_classifier/main.py
```

To run the model on different text data, pass a folder using the `-f` argument:

```bash
python src/models/criminal_classifier/main.py -f /path/to/data/folder
```

*Note:* If using a custom data folder, the input dataset must have a column named `quote` containing the text data and `is_criminal` containing the binary classification label.

To run the unit tests (pytest) on this model, execture the following command from the root directory:

```bash
pytest tests/models/test_criminal.py

```

# Emotion Analysis Project


## Structure

- `EmotionAnalyzer`: Runs emotion classification using HuggingFace transformers
- `PlotAnalyzer`: Base class for data preparation and filtering
- `WordCloudGenerator`, `CompositionChart`, `TopEmotionsChart`, `EmotionHist`: Visualization classes

## Usage

### Basic emotion analysis:
```bash
python emotion_analysis.py -f last_words_data_clean.csv -t quote -d date -o output.csv
```

### With specific emotion (e.g., remorse):
```bash
python emotion_analysis.py -f last_words_data_clean.csv -t quote -d date -e remorse -o output.csv
```

### Generate visualizations:
```bash
python emotion_analysis.py -f output.csv -t quote -d date --skip-analysis --wordcloud
python emotion_analysis.py -f output.csv -t quote -d date --skip-analysis --top-emotions
python emotion_analysis.py -f output.csv -t quote -d date --skip-analysis --composition
python emotion_analysis.py -f output.csv -t quote -d date -e remorse --skip-analysis --emotion-hist
python emotion_analysis.py -f output.csv -t quote -d date --skip-analysis --all-plots
```

### Filter by subset:
```bash
python emotion_analysis.py -f output.csv -t quote -d date --skip-analysis --wordcloud --subset criminal
python emotion_analysis.py -f output.csv -t quote -d date --skip-analysis --top-emotions --subset 1900s
```

## Arguments

| Argument | Description |
|----------|-------------|
| `-f`, `--file` | Path to input CSV file (required) |
| `-t`, `--text_col` | Name of text column (required) |
| `-d`, `--date_col` | Name of date column (required) |
| `-o`, `--output` | Path to save processed CSV |
| `-e`, `--emotion` | Specific emotion to analyze (e.g., remorse, joy) |
| `--subset` | Filter: criminal, not_criminal, expected, not_expected, religious, not_religious, pre_1700, 1700s, 1800s, 1900s, 2000s |
| `--skip-analysis` | Skip emotion analysis (use existing columns) |
| `--wordcloud`, `--top-emotions`, `--composition`, `--emotion-hist`, `--all-plots` | Visualization options |

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
│   │   ├── texas_data_processing.ipynb
│   │   ├── texas_label_clean.py
│   │   ├── text_cleaning.ipynb
│   │   └── wikipedia_scraping.ipynb
│   └── models
│       ├── __init__.py
│       ├── criminal_classifier
│       ├── emotion_model
│       └── lda
└── tests
    ├── data
    │   └── test.csv
    └── models
        ├── test_criminal.py
        ├── test_emotion.py
        └── test_lda.py
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


Last Words Data Link: https://drive.google.com/file/d/1bevcP-T7q_yBcauPwTEWG77PcwIanhBY/view?usp=drive_link
