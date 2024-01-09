import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def read_dataset_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df


def train_kmeans(X, n_clusters=2):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X)
    return model


def main():
    csv_file = 'lingspam_dataset.csv'
    df = read_dataset_from_csv(csv_file)
    train_data = df[df['folder'].str.contains('-part[1-9]')]
    test_data = df[df['folder'].str.contains('-part10')]

    X_train, y_train = train_data['message'], train_data['label']
    X_test, y_test = test_data['message'], test_data['label']

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Antrenează K-Means pe setul de date
    kmeans_model = train_kmeans(X_train_tfidf, n_clusters=2)

    # Asociază clusterii la datele de antrenare
    train_clusters = kmeans_model.predict(X_train_tfidf)

    # Evaluează modelul pe setul de testare
    test_clusters = kmeans_model.predict(X_test_tfidf)

    silhouette_train = silhouette_score(X_train_tfidf, train_clusters)
    silhouette_test = silhouette_score(X_test_tfidf, test_clusters)

    print(f"Silhouette Score for Training Data: {silhouette_train}")
    print(f"Silhouette Score for Test Data: {silhouette_test}")

    # Afișează histograma clusterelor pe datele de antrenare
    plt.hist(train_clusters, bins=range(0, len(set(train_clusters)) + 1), align='left', rwidth=0.8)
    plt.title('Distribution of Clusters in Training Data')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Instances')
    plt.show()


if __name__ == "__main__":
    main()
