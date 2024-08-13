import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-interactive plots
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
import io
import base64

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("data.csv")

# Preprocess the data
features = data[['mean_temp', 'max_temp', 'min_temp', 'meanhum', 'meandew', 'pressure']]

# Handling missing data using KNN Imputer
@app.route('/handle_missing_data')
def handle_missing_data():
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = imputer.fit_transform(features)
    X = pd.DataFrame(imputed_data, columns=features.columns)
    return render_template('result.html', title="Missing Data Handling", data=X.to_html(classes='table table-striped'))

# SMOTE for handling imbalanced datasets
@app.route('/handle_imbalanced_data')
def handle_imbalanced_data():
    smote = SMOTE(random_state=42)
    X = features
    X_res, y_res = smote.fit_resample(X, data['heat'])
    X_res = pd.DataFrame(X_res, columns=X.columns)
    return render_template('result.html', title="SMOTE - Handling Imbalanced Data", data=X_res.to_html(classes='table table-striped'))

# K-Means Clustering
@app.route('/kmeans_clustering')
def kmeans_clustering():
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    y_kmeans = kmeans.fit_predict(features)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(features.iloc[:, 0], features.iloc[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
    plt.title("K-Means Clustering")
    plt.xlabel("Mean Temperature")
    plt.ylabel("Max Temperature")

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('plot.html', title="K-Means Clustering", plot_url=plot_url)

# Decision Tree Classifier
@app.route('/decision_tree_classifier')
def decision_tree_classifier():
    X = features
    y = data['heat']
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)

    plt.figure(figsize=(12, 8))
    plot_tree(clf, filled=True, feature_names=X.columns, class_names=np.unique(y).astype(str))
    plt.title("Decision Tree Classifier")

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('plot.html', title="Decision Tree Classifier", plot_url=plot_url)

# Linear Regression
@app.route('/linear_regression')
def linear_regression():
    X = features
    y = data['mean_temp']
    reg = LinearRegression()
    reg.fit(X, y)
    
    plt.figure(figsize=(10, 8))
    y_pred = reg.predict(X)
    plt.scatter(y, y_pred, color='blue', edgecolor='k', s=50, alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Linear Regression Results")

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('plot.html', title="Linear Regression", plot_url=plot_url)

# Voting Classifier
@app.route('/voting_classifier')
def voting_classifier():
    X = features
    y = data['heat']
    
    clf1 = DecisionTreeClassifier(random_state=42)
    clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf3 = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    voting_clf = VotingClassifier(estimators=[('dt', clf1), ('rf', clf2), ('gb', clf3)], voting='soft')
    voting_clf.fit(X, y)
    
    plt.figure(figsize=(10, 8))
    feature_importances = voting_clf.named_estimators_['rf'].feature_importances_
    plt.barh(features.columns, feature_importances, color='skyblue')
    plt.xlabel("Feature Importance")
    plt.title("Voting Classifier - Feature Importances")

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('plot.html', title="Voting Classifier", plot_url=plot_url)

# Naïve Bayes Classifier
@app.route('/naive_bayes_classifier')
def naive_bayes_classifier():
    X = features
    y = data['heat']
    clf = GaussianNB()
    clf.fit(X, y)
    
    y_pred = clf.predict(X)
    accuracy = np.mean(y_pred == y)
    result_text = f"Naïve Bayes Accuracy: {accuracy:.2f}"

    return render_template('result.html', title="Naïve Bayes Classifier", data=result_text)

# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Notes Pages
@app.route('/missing_data_notes')
def missing_data_notes():
    return render_template('missing_data_notes.html')

@app.route('/kmeans_notes')
def kmeans_notes():
    return render_template('kmeans_notes.html')

@app.route('/ml_models_notes')
def ml_models_notes():
    return render_template('ml_models_notes.html')

@app.route('/probabilistic_learning_notes')
def probabilistic_learning_notes():
    return render_template('probabilistic_learning_notes.html')

@app.route('/naive_bayes_notes')
def naive_bayes_notes():
    return render_template('naive_bayes_notes.html')


if __name__ == "__main__":
    app.run(debug=True)
