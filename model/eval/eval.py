from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(class_model, test_df, eval_method):

    # getting true labels
    true = test_df['is_criminal'].values.astype(int)
    # getting predicted labels from CriminalIdentifier class
    pred = class_model.pred_labels(test_df)

     # perform evaluation metric depending on inputted argument
    if eval_method == "accuracy":
        return accuracy_score(true, pred)
    elif eval_method == 'precision':
        return precision_score(true, pred)
    elif eval_method == 'recall':
        return recall_score(true, pred)
    elif eval_method == 'f1':
        return f1_score(true, pred)
    else:
        print("Unknown evaluation method")
        return None
    