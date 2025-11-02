#!/usr/bin/env python

import random as python_random
import json
import argparse
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Input
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, SVMSMOTE

# Make reproducible as much as possible
numpy.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_file", default='data/train.tsv', type=str,
                        help="Input file to learn from (default train_NE.txt)")
    parser.add_argument("-d", "--dev_file", default='data/dev.tsv', type=str,
                        help="Development set (default dev_NE.txt)")
    parser.add_argument("-e", "--embeddings", default='glove.txt', type=str,
                        help="Embedding file we are using (default glove_filtered.json)")
    parser.add_argument("-ts", "--test_file", type=str,
                        help="Separate test set to read from, for which we do not have labels")
    parser.add_argument("-o", "--output_file", type=str,
                        help="Output file to which we write predictions for test set")
    args = parser.parse_args()
    if args.test_file and not args.output_file:
        raise ValueError("Always specify an output file if you specify a separate test set")
    if args.output_file and not args.test_file:
        raise ValueError("Output file is specified but test set is not -- probably you made a mistake")
    return args


def write_to_file(lst, out_file):
    '''Write list to file'''
    with open(out_file, "w", encoding="utf-8") as out_f:
        for line in lst:
            out_f.write(line.strip() + '\n')
    out_f.close()


def read_corpus(corpus_file):
    '''Function to read the corpus file and return the documents and labels
    Input:
        corpus_file: path to the corpus file
        use_sentiment: boolean indicating whether to use sentiment labels (2-class) or category labels (6-class)
    Output:
        documents: list of tokenized documents
        labels: list of labels
    '''

    documents = []
    labels = []
    # Open tsv file
    with open(corpus_file, 'r', encoding='utf-8') as in_file:
        for line in in_file:
            tokens = line.strip().split('\t')
            documents.append(tokens[0])
            labels.append(tokens[1])
    return documents, labels



def read_embeddings(embeddings_file):
    """Read in GloVe embeddings from a text file."""
    embeddings = {}
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    print(f"✅ Loaded {len(embeddings)} embeddings of size {len(next(iter(embeddings.values())))}")
    return embeddings


def vectorizer(sentences, embeddings, embedding_dim=300):
    """Convert a list of tweets (strings) to averaged embedding vectors (one vector per tweet)."""
    vectors = []
    for sent in sentences:
        # simple whitespace tokenizer; you can replace with something fancier
        tokens = [w.strip() for w in sent.split() if w.strip()]
        if not tokens:
            vectors.append(np.zeros(embedding_dim, dtype='float32'))
            continue
        vecs = []
        for token in tokens:
            token_lower = token.lower()
            if token_lower in embeddings:
                vecs.append(embeddings[token_lower])
            else:
                # OOV handling: use zero vector or a small random vector
                vecs.append(np.zeros(embedding_dim, dtype='float32'))
        # average pooling
        mean_vec = np.mean(vecs, axis=0)
        vectors.append(mean_vec)
    return np.vstack(vectors).astype('float32')


def create_model(X_train, Y_train):
    """Create a deeper feed-forward network with dropout and batch norm."""
    n_classes = Y_train.shape[1]
    input_dim = X_train.shape[1]
    print(f"Creating model with input dim {input_dim} and {n_classes} output classes")

    # Choose loss and final activation
    activation_final = "sigmoid"
    loss_function = "binary_crossentropy"


    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        Dense(128, activation='relu'),
        Dropout(0.5),

        Dense(64, activation='relu'),
        Dropout(0.3),

        Dense(n_classes, activation=activation_final)
    ])

    optimizer = Adam(learning_rate=1e-3)

    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=['accuracy']
    )

    return model


def train_model(model, X_train, Y_train, X_dev, Y_dev):
    '''Train the model here'''
    verbose = 1
    epochs = 10
    batch_size = 32

    # Finally fit the model to our data
    model.fit(X_train, Y_train, verbose=verbose, epochs=epochs,
              batch_size=batch_size, validation_data=(X_dev, Y_dev))
    return model


def dev_set_predict(model, X_dev, Y_dev, class_names=None):
    """Do predictions and measure accuracy on labeled dev or test set,
    and print per-class correct/incorrect counts."""
    
    # Predict probabilities
    Y_pred = model.predict(X_dev)
    
    # Convert probabilities -> class labels
    if Y_pred.shape[1] > 1:
        Y_pred_classes = np.argmax(Y_pred, axis=1)
        Y_dev_classes = np.argmax(Y_dev, axis=1)
    else:
        Y_pred_classes = (Y_pred > 0.5).astype(int).flatten()
        Y_dev_classes = Y_dev.flatten()

    # Calculate overall accuracy
    acc = round(accuracy_score(Y_dev_classes, Y_pred_classes), 3)
    print(f"\nAccuracy on dev set: {acc}")

    # Confusion matrix
    cm = confusion_matrix(Y_dev_classes, Y_pred_classes)
    print("\nConfusion Matrix:")
    print(cm)

    # Print per-class correct/incorrect counts
    if class_names is not None:
        print("\nPer-class results:")
        for i, cls in enumerate(class_names):
            correct = cm[i, i]
            incorrect = np.sum(cm[i, :]) - correct
            print(f"  {cls}: Correct = {correct}, Incorrect = {incorrect}")
    else:
        print("\nPer-class results:")
        for i in range(len(cm)):
            correct = cm[i, i]
            incorrect = np.sum(cm[i, :]) - correct
            print(f"  Class {i}: Correct = {correct}, Incorrect = {incorrect}")


def separate_test_set_predict(test_set, embeddings, encoder, model, output_file):
    '''Do prediction on a separate test set for which we do not have a gold standard.
       Write predictions to a file'''
    # Read and vectorize data
    test_emb = vectorizer([x.strip() for x in open(test_set, 'r')], embeddings)
    # Make predictions
    pred = model.predict(test_emb)
    # Convert to numerical labels and back to string labels
    test_pred = numpy.argmax(pred, axis=1)
    labels = [encoder.classes_[idx] for idx in test_pred]
    # Finally write predictions to file
    write_to_file(labels, output_file)


def main():
    '''Main function to train and test neural network given cmd line arguments'''
    args = create_arg_parser()

    # Read in the data and embeddings
    X_train, Y_train = read_corpus(args.train_file)
    X_dev, Y_dev = read_corpus(args.dev_file)
    embeddings = read_embeddings(args.embeddings)

    print("Train label distribution:", Counter(Y_train))
    print("Dev label distribution:  ", Counter(Y_dev))

    # Transform words to embeddings
    X_train_emb = vectorizer(X_train, embeddings)
    X_dev_emb = vectorizer(X_dev, embeddings)

    # Transform string labels to one-hot encodings
    encoder = LabelBinarizer()
    Y_bin_train = encoder.fit_transform(Y_train)
    Y_bin_dev = encoder.transform(Y_dev)

    # Handle class imbalance with SMOTE
    smote = ADASYN(random_state=42)
    X_train_balanced, Y_train_balanced = smote.fit_resample(X_train_emb, Y_bin_train)

    print("Before SMOTE:", np.bincount(Y_bin_train.flatten()))
    print("After SMOTE:", np.bincount(Y_train_balanced.flatten()))

    # Ensure Y is 2D for Keras (important!)
    if Y_train_balanced.ndim == 1:
        Y_train_balanced = Y_train_balanced.reshape(-1, 1)
    if Y_bin_dev.ndim == 1:
        Y_bin_dev = Y_bin_dev.reshape(-1, 1)

    print("Classes:", encoder.classes_)
    print("Y_train_balanced shape:", Y_train_balanced.shape)


    # Create model
    model = create_model(X_train_balanced, Y_train_balanced)

    # Train the model
    model = train_model(model, X_train_balanced, Y_train_balanced, X_dev_emb, Y_bin_dev)

    # Calculate accuracy on the dev set
    dev_set_predict(model, X_dev_emb, Y_bin_dev)

    # If we specified a test set, there are no gold labels available
    # Do predictions and print them to a separate file
    if args.test_file:
        separate_test_set_predict(args.test_file, embeddings, encoder, model, args.output_file)


if __name__ == '__main__':
    main()
