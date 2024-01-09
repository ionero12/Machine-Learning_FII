import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


def read_dataset_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df


def train_knn(x_train, y_train, n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(x_train, y_train)
    return model


def main():
    csv_file = 'lingspam_dataset.csv'

    df = read_dataset_from_csv(csv_file)
    train_data = df[df['folder'].str.contains('-part[1-9]')]

    test_data = df[df['folder'].str.contains('-part10')]

    X_train, y_train = train_data['message'], train_data['label']
    X_test, y_test = test_data['message'], test_data['label']

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    for n_neighbors in range(10, 500, 10):
        model = train_knn(X_train_tfidf.toarray(), y_train, n_neighbors)
        y_pred = model.predict(X_test_tfidf.toarray())
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for K={n_neighbors}: {accuracy}")


if __name__ == "__main__":
    main()
