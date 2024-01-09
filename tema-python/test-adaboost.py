import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report


def read_dataset_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df


def train_adaboost(X_train, y_train, n_estimators=50, learning_rate=1.0):
    model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
    model.fit(X_train, y_train)
    return model


def main():
    csv_file = 'lingspam_dataset.csv'
    df = read_dataset_from_csv(csv_file)

    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = train_adaboost(X_train_tfidf.toarray(), y_train)

    y_pred = model.predict(X_test_tfidf.toarray())
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)


if __name__ == "__main__":
    main()
