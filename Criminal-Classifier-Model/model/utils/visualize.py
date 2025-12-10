import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# utility function used to plot the top n_features, using analyze_results()
def plot_features(class_identifier, num_feat, label):
    # same process as analyze_results function from class_model.py
    coefs = class_identifier.logistic_model.coef_[0]
    features = class_identifier.feature_names_

    # plot results depending on criminal or not, getting index of these features
    if label == "criminal":
        idx = np.argsort(coefs)[-num_feat:][::-1]
        color = "#00274c"
    elif label == "noncriminal":
        idx = np.argsort(coefs)[:num_feat]
        color = "#FFCB05"
    else:
        raise ValueError("Incorrectly specified label")
    
    # get words from important coefficient indexes 
    top_feats = [features[i] for i in idx]
    top_coefs = coefs[idx]

    # set up bar plots for top features, depending on label argument
    fig, ax = plt.subplots(figsize = (10, 8))
    ax.barh(range(num_feat), top_coefs, color = color, alpha = 0.8)
    ax.set_yticks(range(num_feat))
    ax.set_yticklabels(top_feats)
    ax.set_xlabel(f"Coefficient for {label}", fontsize = 12)
    ax.set_title(f"Top Predictors for {label} label", fontsize = 15, fontweight = 'bold')
    ax.grid(axis = 'x')
    plt.tight_layout()

    # save image
    filepath = 'feature_importance.png'
    typepath = "plots/" + label + "_"
    plt.savefig(typepath + filepath, dpi = 300, bbox_inches = 'tight')
    plt.show()
    
# utility function to plot confusion matrix 
def correlation_matrix(class_identifier, test_df):
    # getting true and predicted labels
    true = test_df['is_criminal'].values.astype(int)
    pred = class_identifier.pred_labels(test_df)

    # calculating confusion matrix using object
    confus_mat = confusion_matrix(true, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix = confus_mat, display_labels = ['Non-Criminal', 'Criminal'])

    # plotting
    fig, ax = plt.subplots(figsize = (8, 6))
    disp.plot(ax = ax, cmap = 'Blues', values_format = 'd', colorbar = True)
    ax.set_title('Confusion Matrix', fontweight = 'bold', fontsize = 14)
    ax.set_xlabel('Predicted Label', fontsize = 12)
    ax.set_ylabel('True Label', fontsize = 12)
    plt.tight_layout()

    # save image
    plt.savefig('plots/confusion_matrix.png', dpi = 300, bbox_inches = 'tight')
    plt.show()
