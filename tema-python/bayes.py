from itertools import cycle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer


class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_probs = {}
        self.word_probs = {}
        self.classes = []

    def fit(self, X, y):
        self.classes = np.unique(y)

        for c in self.classes:
            self.class_probs[c] = (y == c).sum() / len(y)

        vectorizer = CountVectorizer()
        X_count = vectorizer.fit_transform(X)
        feature_names = vectorizer.get_feature_names_out()

        for i, c in enumerate(self.classes):
            class_word_counts = X_count[y == c].sum(axis=0)
            total_words_in_class = class_word_counts.sum()

            self.word_probs[c] = {}
            for j, word in enumerate(feature_names):
                word_prob = (class_word_counts[0, j] + self.alpha) / (
                        total_words_in_class + self.alpha * len(feature_names))
                self.word_probs[c][word] = word_prob

    def predict(self, X):
        predictions = []

        for x in X:
            probs = {c: np.log(self.class_probs[c]) for c in self.classes}

            for word in x.split():
                for c in self.classes:
                    if word in self.word_probs[c]:
                        probs[c] += np.log(self.word_probs[c][word])

            predicted_class = max(probs, key=probs.get)
            predictions.append(predicted_class)

        return predictions


def read_dataset_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df


def leave_one_out_cross_validation(X, y, model):
    accuracies = []

    for i in range(len(X)):
        train_mask = np.ones(len(X), dtype=bool)
        train_mask[i] = False

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[i]
        y_test = y[i]

        model.fit(X_train, y_train)
        y_pred = model.predict([X_test])

        accuracy = accuracy_score([y_test], y_pred)
        accuracies.append(accuracy)
        print(f"Finished iteration {i + 1} out of {len(X)}")

    return accuracies


def test_cvloo(X_data, y_data):
    naive_bayes = NaiveBayes(alpha=1.0)
    accuracies_nb_loo = leave_one_out_cross_validation(X_data, y_data, naive_bayes)

    plt.figure(figsize=(10, 5))
    plt.plot(accuracies_nb_loo, marker='o', linestyle='-', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Leave-One-Out Cross-Validation Accuracy')
    plt.show()

    print("\nNaive Bayes Results:")
    print(f"Average Leave-One-Out Cross-Validation Accuracy: {np.mean(accuracies_nb_loo):.4f}")


def test_bayes(X_train, y_train, X_test, y_test):
    naive_bayes = NaiveBayes(alpha=1.0)
    naive_bayes.fit(X_train, y_train)
    y_pred_nb = naive_bayes.predict(X_test)

    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    report_nb = classification_report(y_test, y_pred_nb)

    print("\nNaive Bayes Results:")
    print(f"Accuracy: {accuracy_nb}")
    print("Classification Report:")
    print(report_nb)

    return accuracy_nb


def main():
    csv_file = 'lingspam_dataset.csv'
    df = read_dataset_from_csv(csv_file)

    train_data = df[df['folder'].str.contains('-part[1-9]')]
    test_data = df[df['folder'].str.contains('-part10')]

    X_train, y_train = train_data['message'], train_data['label']
    X_test, y_test = test_data['message'], test_data['label']
    X_data, y_data = df['message'].values, df['label'].values

    accuracy = test_bayes(X_train, y_train, X_test, y_test)

    plt.figure(figsize=(8, 5))
    plt.bar(['Naive Bayes'], [accuracy], color='blue')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Naive Bayes Test Accuracy')
    plt.show()

    # Test cvloo - this takes 8 hours :)
    # test_cvloo(X_data, y_data)


if __name__ == "__main__":
    main()
