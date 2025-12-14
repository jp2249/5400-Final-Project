# need to import packages
import argparse
from pathlib import Path
# importing all created packages
from model.utils.load_data import csv_df, train_test_df
from model.utils.visualize import plot_features, correlation_matrix
from model.class_model import CriminalIdentifier
from model.eval.eval import evaluate


# importing for logging and setting up
import logging


logging.basicConfig(level = logging.INFO, format = "%(message)s")
logger = logging.getLogger(__name__)


# run command: python main.py -f path/to/raw/data
# to run the entire pipeline


def main(folder_path):
    logger.info("Starting data processing pipeline")
    # reading in data
    logger.info("Reading in raw data")
    data = csv_df(folder_path)


    logger.info("Cleaning and splitting data into training and testing splits")
    # splitting data and saving to new folder
    train_df, test_df = train_test_df(data)
    # saving proessed file
    output_dir_train = Path("data/processed_data")
    output_dir_test = Path("tests/data")
    
    # Ensure directories exist
    output_dir_train.mkdir(parents = True, exist_ok = True)
    output_dir_test.mkdir(parents = True, exist_ok = True)


    train_df.to_csv(output_dir_train / "train.csv", index=False)
    test_df.to_csv(output_dir_test / "test.csv", index=False)
    logger.info(f"Saved training data to {output_dir_train}")
    logger.info(f"Saved testing data to {output_dir_test}")


    # training model on CriminalIdentifier class
    logger.info("Training CriminalIdentifier model")
    class_identifier = CriminalIdentifier()
    class_identifier.train_df(train_df)


    # print("Making predictions")
    # print(f"TEST QUOTES:")
    # print(test_df.head())
    # sample_predictions = class_identifier.pred_labels(test_df.head())
    # sample_probs = class_identifier.pred_probabilities(test_df.head())


    logger.info("Evaluating model predictions")
    # now looking at results by all evaluation metrics
    # getting predictions from test_df from eval.py fle
    exact_metrics = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        score = evaluate(class_identifier, test_df, metric)
        exact_metrics[metric] = score
        logger.info(f"{metric.capitalize()}: {score:.4f}")


    logger.info(f"Final evaluation metrics: {exact_metrics}")


    logger.info("Analyzing top features")
    top_features = class_identifier.analyze_results(num_feat = 10)
    logger.info(f"Top features: {top_features}")


    logger.info("Plotting feature importance for criminal class")
    plot_features(class_identifier, label = "criminal", num_feat = 10)
    logger.info("Plotting feature importance for non-criminal class")
    plot_features(class_identifier, label = "noncriminal", num_feat = 10)


    logger.info("Plotting correlation matrix")
    correlation_matrix(class_identifier, test_df)


    # finished
    logger.info("Pipeline completed successfully")


# actually running main function
if __name__ == "__main__":
    # using argparse for user-end API ease
    parser = argparse.ArgumentParser(description = "Load and preprocess last words data")
    # add flag to pass through different raw files to apply project on 
    # call by python main.py -f /data/path
    parser.add_argument("-f", "--folder", default = "data/processed_data", help = "Path to raw data folder")
    args = parser.parse_args()
    
    main(args.folder)