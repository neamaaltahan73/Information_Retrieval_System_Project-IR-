import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from Final import tfidf_matrix, processed_texts


# Elbow Method لتحديد العدد الأمثل من المجموعات
def determine_optimal_clusters(tfidf_matrix, max_clusters=10):
    inertia = []
    silhouette_scores = []
    for n in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(tfidf_matrix)
        inertia.append(kmeans.inertia_)
        score = silhouette_score(tfidf_matrix, kmeans.labels_)
        silhouette_scores.append(score)
    return inertia, silhouette_scores

# تابع لتجميع الوثائق باستخدام K-Means
def cluster_documents(tfidf_matrix, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(tfidf_matrix)
    clusters = kmeans.labels_
    return clusters, kmeans

# تابع لعرض نتائج التجميع
def display_clusters(processed_texts, clusters, num_clusters):
    clustered_texts = {i: [] for i in range(num_clusters)}
    for text, cluster_id in zip(processed_texts, clusters):
        clustered_texts[cluster_id].append(text)

    for cluster_id, texts in clustered_texts.items():
        print(f"Cluster {cluster_id}:")
        for text in texts[:10]:  # عرض أول 10 وثائق فقط لكل مجموعة
            print(f"- {text}")
        print("\n")

# تابع لرسم التجمعات باستخدام PCA لتقليل الأبعاد
def plot_clusters(tfidf_matrix, clusters, num_clusters):
    pca = PCA(n_components=2)
    reduced_tfidf = pca.fit_transform(tfidf_matrix.toarray())
    plt.figure(figsize=(10, 8))
    for cluster_id in range(num_clusters):
        points = reduced_tfidf[clusters == cluster_id]
        plt.scatter(points[:, 0], points[:, 1], label=f'Cluster {cluster_id}')
    plt.legend()
    plt.title("Document Clusters")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()

# استخدام Elbow Method لتحديد العدد الأمثل من المجموعات
max_clusters = 10
inertia, silhouette_scores = determine_optimal_clusters(tfidf_matrix, max_clusters)

# تأكد من أن القيم ليست فارغة
if not inertia or not silhouette_scores:
    raise ValueError("Inertia or silhouette scores are empty. Check your tfidf_matrix and clustering logic.")

# طباعة القيم للتحقق منها
print("Inertia values:", inertia)
print("Silhouette scores:", silhouette_scores)

# رسم منحنى Elbow Method
plt.figure(figsize=(10, 8))
plt.plot(range(2, max_clusters + 1), inertia, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# رسم منحنى Silhouette Scores
plt.figure(figsize=(10, 8))
plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
plt.title('Silhouette Scores for Optimal Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()

# اختيار العدد الأمثل للمجموعات بناءً على Elbow Method و Silhouette Scores
optimal_clusters = 5  # حدد العدد الأمثل بناءً على التحليل السابق

# تجميع الوثائق وعرض النتائج
clusters, kmeans_model = cluster_documents(tfidf_matrix, optimal_clusters)
display_clusters(processed_texts, clusters, optimal_clusters)
plot_clusters(tfidf_matrix, clusters, optimal_clusters)




import os
import re
import json
import string
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import nltk
import spacy
import csv
import pickle

# Load NLTK and Spacy libraries
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")

# Function to process text
def execute_operations_on_text(text):
    lowercased = text.lower().translate(str.maketrans('', '', string.punctuation))
    doc = nlp(lowercased)
    no_stopwords = ' '.join([token.text for token in doc if not token.is_stop])
    no_numbers = re.sub(r'\d+', '', no_stopwords)
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    words = word_tokenize(no_numbers)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    stemmed_words = [stemmer.stem(word) for word in lemmatized_words]
    no_whitespace = ' '.join(stemmed_words).strip()
    return no_whitespace

# Function to create acronym dictionary
def create_acronym_dic(acronymFileName):
    acronym = {}
    try:
        with open(acronymFileName, encoding="utf8") as file:
            csv_file = csv.reader(file, delimiter="\t")
            for line in csv_file:
                acronym[line[0]] = line[1]
        print("Acronym dictionary created successfully.")
    except Exception as e:
        print(f"Error reading acronym file: {e}")
    return acronym

# Function to replace acronyms in texts
def replace_acronym(text, mydict):
    words = text.split()
    replaced_text = ' '.join([mydict.get(word, word) for word in words])
    return replaced_text

# Function to prepare dataset and process texts
def prepare_data(file_path, acronymFileName, num_lines=20000):
    acronym_dic = create_acronym_dic(acronymFileName)
    processed_texts = []
    titles = []
    bodies = []
    with open(file_path, "rt", encoding="utf8") as fin:
        for i, line in enumerate(fin):
            if i < num_lines:
                title, body = line.strip().split('\t')
                processed_title = execute_operations_on_text(title)
                processed_body = execute_operations_on_text(body)
                processed_title = replace_acronym(processed_title, acronym_dic)
                processed_body = replace_acronym(processed_body, acronym_dic)
                processed_texts.append((processed_title, processed_body))
                titles.append(processed_title)
                bodies.append(processed_body)
            else:
                break
    return processed_texts, titles, bodies

# Function to create TF-IDF matrix
def create_tfidf_matrix(processed_texts):
    vectorizer = TfidfVectorizer()
    combined_texts = [' '.join(text) for text in processed_texts]
    tfidf_matrix = vectorizer.fit_transform(combined_texts)
    return tfidf_matrix, vectorizer

# Function to process query
def process_query(query, vectorizer):
    processed_query = execute_operations_on_text(query)
    query_vector = vectorizer.transform([processed_query])
    return query_vector

# Function to compute similarity using Cosine Similarity
def compute_similarity(query_vector, tfidf_matrix):
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)
    return cosine_similarities

# Function to search the index
def search_index(query, vectorizer, tfidf_matrix, processed_texts):
    query_vector = process_query(query, vectorizer)
    cosine_similarities = compute_similarity(query_vector, tfidf_matrix)
    sorted_indices = cosine_similarities[0].argsort()[::-1]
    results = [(index, cosine_similarities[0][index], processed_texts[index][1]) for index in sorted_indices]
    return results

# Function to calculate evaluation metrics
def calculate_evaluation_metrics(evaluation_file_path, vectorizer, tfidf_matrix, processed_texts):
    with open(evaluation_file_path, 'r', encoding='utf-8') as evaluation_file:
        data = [json.loads(line.strip()) for line in evaluation_file]

    relevant_docs = defaultdict(list)
    for item in data:
        relevant_docs[item['qid']].extend(item['answer_pids'])

    precision_list = []
    recall_list = []
    average_precision_sum = 0
    reciprocal_rank_sum = 0

    for query_index, query_data in enumerate(data):
        if query_index >= 10:
            break

        query_id = query_data['qid']
        query_text = query_data['query']
        relevant_pids = query_data['answer_pids']

        search_results = search_index(query_text, vectorizer, tfidf_matrix, processed_texts)
        retrieved_pids = [result[0] for result in search_results]

        relevant_retrieved = len(set(relevant_pids) & set(retrieved_pids))
        precision = relevant_retrieved / len(retrieved_pids) if len(retrieved_pids) > 0 else 0
        recall = relevant_retrieved / len(relevant_pids) if len(relevant_pids) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)

        average_precision = 0
        relevant_count = 0
        for i, pid in enumerate(retrieved_pids):
            if pid in relevant_pids:
                relevant_count += 1
                average_precision += relevant_count / (i + 1)
        average_precision /= len(relevant_pids) if len(relevant_pids) > 0 else 1
        average_precision_sum += average_precision

        for i, pid in enumerate(retrieved_pids):
            if pid in relevant_pids:
                reciprocal_rank_sum += 1 / (i + 1)
                break

        print(f"Query {query_id}: {query_text}")
        print(f"Relevant PIDs: {relevant_pids}")
        print(f"Retrieved PIDs: {retrieved_pids[:10]}")
        for idx, (index, score, text) in enumerate(search_results[:10]):
            print(f"Result {idx+1}: DocID = {index}, Score = {score}, Text = {text}")
        print("\n")

    query_count = min(len(data), 10)
    map_score = average_precision_sum / query_count if query_count > 0 else 0.0
    mrr_score = reciprocal_rank_sum / query_count if query_count > 0 else 0.0

    evaluation_metrics = {
        'MAP': map_score,
        'MRR': mrr_score,
        'Precision': precision_list,
        'Recall': recall_list
    }
    return evaluation_metrics

# Prepare dataset and create TF-IDF matrix
folder = "C:/Users/lenovo/Desktop/ir-pro/lotte/lifestyle/dev/"
acronymFileName = os.path.join(folder, "acrony.csv")
file_path = os.path.join(folder, "collection.tsv")
processed_texts, processed_titles, processed_bodies = prepare_data(file_path, acronymFileName)

if not processed_texts:
    print("No documents found. Please check the dataset file.")
else:
    tfidf_matrix, vectorizer = create_tfidf_matrix(processed_texts)

    # Save TF-IDF matrix and vectorizer to files
    tfidf_matrix_file = os.path.join(folder, "tfidf_matrix.pkl")
    vectorizer_file = os.path.join(folder, "vectorizer.pkl")
    with open(tfidf_matrix_file, 'wb') as fout:
        pickle.dump(tfidf_matrix, fout)
    with open(vectorizer_file, 'wb') as fout:
        pickle.dump(vectorizer, fout)

    # Load TF-IDF matrix and vectorizer from files
    with open(tfidf_matrix_file, 'rb') as fin:
        tfidf_matrix = pickle.load(fin)
    with open(vectorizer_file, 'rb') as fin:
        vectorizer = pickle.load(fin)

    # Calculate evaluation metrics
    evaluation_file_path = os.path.join(folder, "qas.search.jsonl")
    evaluation_metrics = calculate_evaluation_metrics(evaluation_file_path, vectorizer, tfidf_matrix, processed_texts)

    # Print results
    print(f"Processed Texts: {processed_texts[:5]}")
    print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")

    # Print Mean Average Precision (MAP) and Mean Reciprocal Rank (MRR)
    print(f"Mean Average Precision (MAP): {evaluation_metrics['MAP']}")
    print(f"Mean Reciprocal Rank (MRR): {evaluation_metrics['MRR']}")

    for i, (precision, recall) in enumerate(zip(evaluation_metrics['Precision'][:10], evaluation_metrics['Recall'][:10])):
        print(f"Query {i+1}: Precision = {precision}, Recall = {recall}")

