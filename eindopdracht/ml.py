#!/usr/bin/env python

'''
Script name: lfd_assignment1.py
Author: Timo Strijbis

Usage: python ml.py -h'''

#import required libraries
import argparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import nltk
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
nltk.download('averaged_perceptron_tagger_eng') 
from sklearn.preprocessing import StandardScaler
import numpy as np
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, SVMSMOTE


def create_arg_parser():
    '''Function to create the argument parser'''
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_file", default='data/train.tsv', type=str,
                        help="Train file to learn from (default data/train.tsv)")
    parser.add_argument("-d", "--dev_file", default='data/dev.tsv', type=str,
                        help="Dev file to evaluate on (default data/dev.tsv)")
    parser.add_argument("-tf", "--tfidf", action="store_true", default=True,
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
    parser.add_argument("-svm", "--svm", action="store_true",
                        help="Use the Support Vector Machine classifier")
    parser.add_argument("-lr", "--logistic_regression", action="store_true",
                        help="Use the Logistic Regression classifier")
    parser.add_argument("-rf", "--random_forest", action="store_true",
                        help="Use the Random Forest classifier")
    parser.add_argument("-knn", "--k_nearest_neighbors", action="store_true",
                        help="Use the K-Nearest Neighbors classifier")
    parser.add_argument("-s", "--smote", action="store_true",
                        help="Use SMOTE for handling class imbalance")
    parser.add_argument("-bs", "--border_smote", action="store_true",
                        help="Use Borderline-SMOTE for handling class imbalance")
    parser.add_argument("-a", "--adasyn", action="store_true",
                        help="Use ADASYN for handling class imbalance")
    parser.add_argument("-sv", "--svm_smote", action="store_true",
                        help="Use SVM-SMOTE for handling class imbalance")
    args = parser.parse_args()
    return args


def read_corpus(corpus_file):
    '''Function to read the corpus file and return the documents and labels'''

    documents = []
    labels = []
    # Open tsv file
    with open(corpus_file, 'r', encoding='utf-8') as in_file:
        for line in in_file:
            tokens = line.strip().split('\t')
            documents.append(tokens[0])
            labels.append(tokens[1])
    return documents, labels


def identity(inp):
    '''Dummy function that just returns the input'''
    return inp


if __name__ == "__main__":
    args = create_arg_parser()

    # Read the train and test files
    X_train, Y_train = read_corpus(args.train_file)
    X_test, Y_test = read_corpus(args.dev_file)


    # Choose vectorizer based on argument
    if args.tfidf:
        print("Using TF-IDF vectorizer")
        vectorizer = TfidfVectorizer(preprocessor=identity, tokenizer=identity)

    # If not using TF-IDF, use CountVectorizer
    else:
        print("Using Count vectorizer")
        vectorizer = CountVectorizer(preprocessor=identity, tokenizer=identity)

    # Vectorize training and test data
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"Original training set distribution: {np.unique(Y_train, return_counts=True)[1][0]}, {np.unique(Y_train, return_counts=True)[1][1]}")

    # --- Apply selected oversampling method ---
    if args.smote:
        print("Applying regular SMOTE...")
        sampler = SMOTE(random_state=42)
    elif args.border_smote:
        print("Applying Borderline-SMOTE...")
        sampler = BorderlineSMOTE(random_state=42, kind='borderline-1')
    elif args.adasyn:
        print("Applying ADASYN...")
        sampler = ADASYN(random_state=42)
    elif args.svm_smote:
        print("Applying SVM-SMOTE...")
        sampler = SVMSMOTE(random_state=42)
    else:
        sampler = None

    if sampler:
        X_train_vec, Y_train = sampler.fit_resample(X_train_vec, Y_train)
        print(f"Distribution after resampling using {sampler.__class__.__name__}: {np.unique(Y_train, return_counts=True)[1][0]}, {np.unique(Y_train, return_counts=True)[1][1]}")


    # Classifier selection
    if args.logistic_regression:
        print("Using Logistic Regression classifier")
        classifier = LogisticRegression(max_iter=1000)
    elif args.k_nearest_neighbors:
        print("Using K-Nearest Neighbors classifier")
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier()
    elif args.random_forest:
        print("Using Random Forest classifier")
        classifier = RandomForestClassifier()
    elif args.svm:
        print("Using Support Vector Machine classifier")
        classifier = SVC()
    else:
        print("No classifier option provided, using Random Forest by default")
        classifier = RandomForestClassifier()
    

    # Train classifier
    classifier.fit(X_train_vec, Y_train)

    # Predict on test set
    Y_pred = classifier.predict(X_test_vec)

    # Compute metrics
    acc = accuracy_score(Y_test, Y_pred)
    prec = precision_score(Y_test, Y_pred, average='weighted')
    rec = recall_score(Y_test, Y_pred, average='weighted')
    f1 = f1_score(Y_test, Y_pred, average='weighted')

    print(f"\nFinal accuracy: {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall: {rec:.2f}")
    print(f"F1 score: {f1:.2f}")

    # --- Compute per-class correct/incorrect counts and accuracy ---
    classes = np.unique(Y_test)
    print("\nPer-class results:")
    for cls in classes:
        indices = [i for i, y in enumerate(Y_test) if y == cls]
        total_cls = len(indices)
        correct_cls = sum(1 for i in indices if Y_test[i] == Y_pred[i])
        incorrect_cls = total_cls - correct_cls
        cls_acc = correct_cls / total_cls
        print(f"Class '{cls}': Correct = {correct_cls}, Incorrect = {incorrect_cls}, Accuracy = {cls_acc:.2f}")