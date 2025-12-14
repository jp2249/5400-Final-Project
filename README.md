# 5400-Final-Project

### Team Members:

- Jeffrey Pinarchick
- Tyler McCormick
- Younghoon Kim
- Sam Gold
- Nikhil Poluri

# Project aim and research questions

- How Do Humans comprehend Death?
- What do people say in their last words?
- If you knew that you were going to die, what would you say?
- What is your last word? Who would you think of?

# Tree of the whole project

```text
.
├── criminal_classifier
│   ├── main.py
│   ├── model
│   └── run_code.txt
├── data
│   ├── processed_data
│   └── raw_data
├── data_process
│   ├── __init__.py
│   ├── texas_data_processing.ipynb
│   ├── texas_data_processing.py
│   ├── wikipedia_scraping.ipynb
│   └── wikipedia_scraping.py
├── Emotion-Model
│   ├── __init__.py
│   └── emotion_analysis.py
├── environment.yml
├── lda
│   ├── __init__.py
│   ├── lda_last_words_kgrid.py
│   └── lda_last_words.py
├── pyproject.toml
├── pytest.ini
├── README.html
├── README.md
└── tests
    ├── data
    └── models
```

# Setup and Installation

This project uses **Conda** for environment management and **pip** for package installation. The code is organized in the **root** directory.

## Prerequisites

- **Anaconda or Miniconda**: [Installation Guide](https://docs.anaconda.com/free/miniconda/)

## Installation Steps

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

# Data Processing

## Collect raw data from Wikipedia and Texas execution data
```bash
python data_process/wikipedia_scraping.py
python data_process/texas_data_processing.py
```

## Access to processed data

Since the raw data requires manual cleaning and manual labeling, we provide the processed data here:

https://drive.google.com/file/d/1bevcP-T7q_yBcauPwTEWG77PcwIanhBY/view?usp=drive_link

Place the processed data in the `data/processed_data` directory.


# Testing all three models
To run the full test suite (covering Criminal Identifier, Emotion Analysis, and LDA), execute the following command from the root directory:

```bash
python -m pytest tests/
```


# The Criminal Identifier Model

## Model description

The Criminal Identifier model is a binary classification system designed to determine whether a given quote belongs to a criminal or a non-criminal. It utilizes natural language processing (NLP) techniques to analyze the text and a machine learning classifier to make predictions.


## Running the Criminal Identifier model

To run the entire workflow on the original data, execute the following command from the root directory:

```bash
python criminal_classifier/main.py
```

To run the model on different text data, pass a folder using the `-f` argument:

```bash
python criminal_classifier/main.py -f /path/to/data/folder
```

*Note:* If using a custom data folder, the input dataset must have a column named `quote` containing the text data and `is_criminal` containing the binary classification label.


# Emotion Analysis Model

## Structure

- `EmotionAnalyzer`: Runs emotion classification using HuggingFace transformers
- `PlotAnalyzer`: Base class for data preparation and filtering
- `WordCloudGenerator`, `CompositionChart`, `TopEmotionsChart`, `EmotionHist`: Visualization classes

## Usage

### Basic emotion analysis:
```bash
python Emotion-Model/emotion_analysis.py -f data/processed_data/last_words_data.csv -t quote -d date -o data/processed_data/emotion_output.csv
```

### With specific emotion (e.g., remorse):
```bash
python Emotion-Model/emotion_analysis.py -f data/processed_data/last_words_data.csv -t quote -d date -e remorse -o data/processed_data/emotion_output.csv
```

### Generate visualizations:
```bash
python Emotion-Model/emotion_analysis.py -f data/processed_data/emotion_output.csv -t quote -d date --skip-analysis --wordcloud
python Emotion-Model/emotion_analysis.py -f data/processed_data/emotion_output.csv -t quote -d date --skip-analysis --top-emotions
python Emotion-Model/emotion_analysis.py -f data/processed_data/emotion_output.csv -t quote -d date --skip-analysis --composition
python Emotion-Model/emotion_analysis.py -f data/processed_data/emotion_output.csv -t quote -d date -e remorse --skip-analysis --emotion-hist
python Emotion-Model/emotion_analysis.py -f data/processed_data/emotion_output.csv -t quote -d date --skip-analysis --all-plots
```

### Filter by subset:
```bash
python Emotion-Model/emotion_analysis.py -f data/processed_data/emotion_output.csv -t quote -d date --skip-analysis --wordcloud --subset criminal
python Emotion-Model/emotion_analysis.py -f data/processed_data/emotion_output.csv -t quote -d date --skip-analysis --top-emotions --subset 1900s
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


# LDA Topic Modeling

To uncover latent themes within the last statements, we employ Latent Dirichlet Allocation (LDA). This module provides two scripts for topic modeling:

## Basic LDA Pipeline (`lda/lda_last_words.py`)

This script performs a standard LDA analysis on the entire dataset.

**Pipeline Steps:**
1.  **Preprocessing**: Tokenization, lemmatization, and stopword removal using spaCy and NLTK.
2.  **Bigram Generation**: Automatically detects and forms common phrases (e.g., "death row", "heavenly father").
3.  **Topic Modeling**: Trains LDA models with varying numbers of topics (K=4, 6, 8).
4.  **Coherence Tuning**: Selects the best K based on Coherence Score (c_v).

**Usage:**
```bash
python lda/lda_last_words.py --csv data/processed_data/last_words_data.csv
```

## Advanced Segmented Analysis (`lda/lda_last_words_kgrid.py`)

This script performs a more granular analysis by running LDA on specific subsets of the data (e.g., per century, by criminal status) with adaptive hyperparameter tuning.

**Features:**
-   **Automatic Year Extraction**: Parses diverse date formats to bucket statements by century.
-   **Adaptive K-Grid**: Dynamically adjusts the grid of K values (number of topics) based on the size of the subset (number of documents) to avoid overfitting small groups.
-   **Segmented Execution**: Automatically runs separate models for:
    -   The entire dataset
    -   Each Century (17th - 21st)
    -   Criminal vs. Non-Criminal
    -   Religious vs. Non-Religious

**Usage:**
```bash
python lda/lda_last_words_kgrid.py --csv data/processed_data/last_words_data.csv
```



# Project Status

## Project Structure & Setup
- [x] Initialize with `pyproject.toml` (not setup.py)
- [x] Create `environment.yml` file with all dependencies
- [x] Set up proper directory structure (src/, tests/, data/, docs/, etc.)

## README Documentation
- [x] List full names of all team members
- [x] Project aim and research question
- [x] Installation instructions
- [x] Usage examples with code snippets
- [ ] Architecture diagram (created with draw.io or similar)
- [x] Data download/access instructions
- [x] Dependencies and environment setup guide

## Data Management
- [x] Set up data storage (GitHub LFS or Google drive link)
- [x] Create data download script if possible
- [ ] Implement data preprocessing pipeline
- [x] Store preprocessed data in folder
- [x] Document data sources and preprocessing steps
- [x] NO data pushed to GitHub

## Code Implementation
- [ ] Design OOP (classes for model, preprocessing, evaluation, etc.)
- [ ] Add comments to all functions, methods, and classes
- [ ] Implement logging at key points (data loading, training, evaluation)
- [ ] Follow PEP 8 standards using pylint (clean structure)
 
## Testing Suite
- [x] Create `tests/` directory with pytest structure
- [x] Set separate test dataset
- [ ] Write tests for model components
- [ ] All tests pass d


 