# need to import packages
import argparse
from pathlib import Path
# importing all created packages
from model.utils.load_data import csv_df, train_test_df
from model.utils.visualize import plot_features, correlation_matrix
from model.class_model import CriminalIdentifier
from model.eval.eval import evaluate

# run command: python main.py -f path/to/raw/data
# to run the entire pipeline

def main(folder_path):
    # reading in data
    print("Readng in the raw data:")
    data = csv_df(folder_path)

    print("Cleaning and splitting data into training and testing splits")
    # splitting data and saving to new folder
    train_df, test_df = train_test_df(data)
    # saving proessed file
    output_dir_train = Path("data/processed_data")
    output_dir_test = Path("tests/data")
    
    # Ensure directories exist
    output_dir_train.mkdir(parents=True, exist_ok=True)
    output_dir_test.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir_train / "train.csv", index=False)
    test_df.to_csv(output_dir_test / "test.csv", index=False)
    print(f"Data splits saved: Train -> {output_dir_train}, Test -> {output_dir_test}")

    # training model on CriminalIdentifier class
    print("Training model")
    class_identifier = CriminalIdentifier()
    class_identifier.train_df(train_df)

    print("Making predictions")
    # print(f"TEST QUOTES:")
    # print(test_df.head())
    # sample_predictions = class_identifier.pred_labels(test_df.head())
    # sample_probs = class_identifier.pred_probabilities(test_df.head())

    print("Evaluating predicted labels")
    # now looking at results by all evaluation metrics
    # getting predictions from test_df from eval.py fle
    exact_metrics = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        score = evaluate(class_identifier, test_df, metric)
        exact_metrics[metric] = score
    print(f"Results: {exact_metrics}")

    print("Checking model's top features")
    top_features = class_identifier.analyze_results(num_feat = 10)

    print("Plotting important features for both classes")
    plot_features(class_identifier, label = "criminal", num_feat = 10)
    plot_features(class_identifier, label = "noncriminal", num_feat = 10)

    print("Plotting correlation matrix")
    correlation_matrix(class_identifier, test_df)

# actually running main function
if __name__ == "__main__":
    # using argparse for user-end API ease
    parser = argparse.ArgumentParser(description = "Load and preprocess last words data")
    # add flag to pass through different raw files to apply project on 
    # call by python main.py -f /data/path
    parser.add_argument("-f", "--folder", default = "data/raw_data", help = "Path to raw data folder")
    args = parser.parse_args()
    
    main(args.folder)