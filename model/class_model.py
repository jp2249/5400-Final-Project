import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd

# moving alpha decreases precision, increases recall
class CriminalIdentifier:
    # initializer
    def __init__(self, alpha = .5, features = 5000, ngram_range = (1, 2)):
        # initializing
        self.alpha = alpha
        self.features = features
        self.ngram_range = ngram_range

        # add other model components that get specified in training
        self.tfidf_vectorizer = None
        self.logistic_model = None
        self.feature_names_ = None
        self.classes_ = None

    # training model, first through tf-idf then logisitc 
    def train_df(self, train_df):
        # creatiing tf-idf object, 1-2 ngrams and english stopwords
        self.tfidf_vectorizer = TfidfVectorizer(max_features = self.features,
                                     ngram_range = self.ngram_range,
                                     lowercase = True,
                                     stop_words = 'english')
        
        # fitting object on training data
        x_train = train_df['quote'].values
        y_train = train_df['is_criminal'].values.astype(int)
        x_train_tfidf = self.tfidf_vectorizer.fit_transform(x_train)
        
        # creating logistic regression clasifier, l2 regularization
        self.logistic_model = LogisticRegression(C = 1 / self.alpha,
                                                 penalty = 'l2',
                                                 max_iter = 1000)
        
        # fitting classifier object on altered tf-idf vectorization training data
        self.logistic_model.fit(x_train_tfidf, y_train)

        self.feature_names_ = self.tfidf_vectorizer.get_feature_names_out()
        self.classes_ = self.logistic_model.classes_

        # for testing purposes
        print(f"Vocabulary size: {len(self.feature_names_)} features")
        print(f"Classes: {self.classes_}")

    # getting predicted class labels, used for evaluation later
    # running same process as train_df for testing data, returning just the predicted labels
    def pred_labels(self, test_df):
        # getting testing data
        x_test = test_df['quote'].values
        x_test_tfidf = self.tfidf_vectorizer.transform(x_test)  

        # making predictions
        predictions = self.logistic_model.predict(x_test_tfidf)
        # print(f"PREDICTIONS: {predictions}")
        return predictions

    # getting predicted class probabilities, used for evaluation later
    # running same process as train_df for testing data, returning just the predicted probabilities
    def pred_probabilities(self, test_df):
         # getting testing data
        x_test = test_df['quote'].values
        x_test_tfidf = self.tfidf_vectorizer.transform(x_test) 

        # getting probabilities estimates
        probabilities = self.logistic_model.predict_proba(x_test_tfidf) 

        print(f"PROBABILITIES: {probabilities}")
        return probabilities
    
    # getting top n important features from modeling for results analysis
    def analyze_results(self, num_feat):
        # getting weights from logisistic regression modeling
        coefs = self.logistic_model.coef_[0]

        # taking top n criminal features, depending on argument
        criminal_idx = np.argsort(coefs)[-num_feat:][::-1]
        for i in criminal_idx:
            print(f"{self.feature_names_[i]}: {coefs[i]:.4f}")
        
        # top n non-criminal features
        non_criminal_idx = np.argsort(coefs)[:num_feat]
        for i in non_criminal_idx:
            print(f"{self.feature_names_[i]}: {coefs[i]:.4f}")
        
        return criminal_idx, non_criminal_idx
