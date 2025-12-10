import pandas as pd
from pathlib import Path
import string
import re
from sklearn.model_selection import train_test_split

# ORIGINAL FUNCTION, HAD TO CHANGE (commented out code at bottom)TO ACCOUNT FOR MISALIGNED COLUMNS
# reading in data, keeping important columns, cleaning data
def csv_df(folder):
    # load in csv and convert to cleaned df
    path = Path(folder) / "last_words_data.csv"
    data = pd.read_csv(path)

    # only selecting necessary columns for modeling
    data_df = data[['quote', 'is_criminal']].copy()

    # removing potential outlier values
    data_df = data_df[data_df["is_criminal"] <= 1].reset_index(drop=True)

    # function to clean text 
    def clean_text(text):
        if pd.isna(text):
            return ""
        # remove non-ASCII ("weird") characters
        text = text.encode('ascii', errors='ignore').decode()

        # removing digits
        # removing puncuation
        clean_text = ""
        for char in text:
            # only keep non-digits or non-puncuation
            if not (char.isdigit() or char in string.punctuation):
                clean_text += char

        # lowercasing
        clean_text = clean_text.lower()

        # fix any whitespace inconsistencies
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        return clean_text

    # applying cleaning function
    data_df['quote'] = data_df['quote'].apply(clean_text)
    # print(data_df['quote'].head())

    # print statements to explore data
    print(f"Dataset shape after cleaning: {data_df.shape}")
    print("\nClass distribution:")
    print(data_df['is_criminal'].value_counts())
    print("\nSample data:")
    print(data_df.head())
    
    return data_df

# splitting data into processed training and testing splits, 80% split
# setting seed for reproducability
def train_test_df(df, train = .8, random_state = 5400):
    # using stratify to keep proportions when splitting 
    groups = df['is_criminal']

    # splitting data 
    train_df, test_df = train_test_split(df, train_size = train, stratify = groups, random_state = random_state)

    # print statements to explore data
    print(train_df.head())
    print(test_df.head())
    print(f"\nTrain shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print("\nTrain class distribution:")
    print(train_df['is_criminal'].value_counts(normalize = True))
    
    return train_df, test_df



################################################################
# Below contains sample code to use for if the columns in the data are all misaligned

# reading in data, keeping important columns, cleaning data
# def csv_df(folder):
#     # load in csv and convert to cleaned df
#     path = Path(folder) / "last_words_data.csv"
#     data = pd.read_csv(path)

#     # setting new column to fix data error
#     data['quote_corrected'] = ""

#     # Rows 0:1114 (1:1115 in 1-based) → quote column
#     # Rows 1115:1468 (1116:1469 in 1-based) → context column 
#     # Rows 1469:2534 (1470:2535 in 1-based) → date column
#     # Rows 2536:end (2537:2941 in 1-based) → quote column again
#     data.iloc[0:1115, data.columns.get_loc('quote_corrected')] = data.iloc[0:1115]['quote']
#     data.iloc[1115:1469, data.columns.get_loc('quote_corrected')] = data.iloc[1115:1469]['context']
#     data.iloc[1469:2536, data.columns.get_loc('quote_corrected')] = data.iloc[1469:2536]['date']
#     data.iloc[2536:, data.columns.get_loc('quote_corrected')] = data.iloc[2536:]['quote']

#     # after cleaning, only choose necessary columns
#     data_df = pd.DataFrame({'quote': data['quote_corrected'],'is_criminal': data['is_criminal']})

#     # removing outlier values
#     data_df = data_df[data_df["is_criminal"] <= 1].reset_index(drop=True)

#     def clean_text(text):
#         if pd.isna(text):
#             return ""
#         # remove non-ASCII ("weird") characters
#         text = text.encode('ascii', errors='ignore').decode()

#         # removing digits
#         # removing puncuation
#         clean_text = ""
#         for char in text:
#             # only keep non-digits or non-puncuation
#             if not (char.isdigit() or char in string.punctuation):
#                 clean_text += char

#         # lowercasing
#         clean_text = clean_text.lower()

#         # fix any whitespace inconsistencies
#         clean_text = re.sub(r"\s+", " ", clean_text).strip()
#         return clean_text

#     data_df['quote'] = data_df['quote'].apply(clean_text)
#     # print(data_df['quote'].head())

#     print(f"Dataset shape after cleaning: {data_df.shape}")
#     print("\nClass distribution:")
#     print(data_df['is_criminal'].value_counts())
#     print("\nSample data:")
#     print(data_df.head())
#     # print(f"CHECKING: {data_df[data_df['quote'] == 'february']}")
    
#     return data_df
################################################################
