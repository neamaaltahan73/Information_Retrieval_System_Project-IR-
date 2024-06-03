import json
import string
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy
import csv
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# تحميل مكتبات NLTK و Spacy
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")

# توابع معالجة النصوص مجمعة في تابع واحد
def execute_operations_on_text(text):
    # تحويل النص إلى أحرف صغيرة وإزالة علامات الترقيم
    lowercased = text.lower().translate(str.maketrans('', '', string.punctuation))
    # إزالة كلمات التوقف باستخدام Spacy
    doc = nlp(lowercased)
    no_stopwords = ' '.join([token.text for token in doc if not token.is_stop])
    # إزالة الأرقام
    no_numbers = re.sub(r'\d+', '', no_stopwords)
    # تقطيع النص إلى كلمات وتطبيق التلميع والتجذير
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    words = word_tokenize(no_numbers)
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    stemmed_words = [stemmer.stem(word) for word in lemmatized_words]
    # إعادة تجميع الكلمات وإزالة المسافات الزائدة
    no_whitespace = ' '.join(stemmed_words).strip()
    return no_whitespace


# تابع لتحضير بيانات مجموعة البيانات ومعالجة النصوص
def prepare_data(file_path, acronymFileName):
    processed_texts = []
    with open(file_path, "rt", encoding="utf8") as fin:
        for line in fin:
            processed_line = execute_operations_on_text(line.strip())
            processed_texts.append(processed_line)
    return processed_texts

# تابع لإنشاء مصفوفة TF-IDF
def create_tfidf_matrix(processed_texts, max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.7):
    vectorizer = TfidfVectorizer(max_features=max_features,
                                 ngram_range=ngram_range,
                                 min_df=min_df,
                                 max_df=max_df)
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    return tfidf_matrix

# تابع لمعالجة الاستعلام
def process_query(query, vectorizer):
    processed_query = execute_operations_on_text(query)
    query_vector = vectorizer.transform([processed_query])
    return query_vector

# تابع لحساب التشابه باستخدام Cosine Similarity
def compute_similarity(query_vector, tfidf_matrix):
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)
    return cosine_similarities

# تابع البحث في الفهرس
def search_index(query, vectorizer, tfidf_matrix, top_n=10):
    query_vector = process_query(query, vectorizer)
    query_words = query.split()  # تقسيم الاستعلام إلى كلمات
    results = []
    for i, doc_vector in enumerate(tfidf_matrix):
        cosine_similarity = cosine_similarity(query_vector, doc_vector)
        if cosine_similarity > 0:  # التأكد من أن هناك تشابه
            results.append((i, cosine_similarity[0][0]))  # إضافة النتيجة إلى القائمة
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)[:top_n]  # فرز النتائج
    return sorted_results


def calculate_evaluation_metrics(evaluation_file_path, vectorizer, tfidf_matrix):
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
        query_id = query_data['qid']
        query_text = query_data['query']
        relevant_pids = query_data['answer_pids']

        search_results = search_index(query_text, vectorizer, tfidf_matrix)
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

    query_count = len(data)
    map_score = average_precision_sum / query_count if query_count > 0 else 0.0
    mrr_score = reciprocal_rank_sum / query_count if query_count > 0 else 0.0

    evaluation_metrics = {
        'MAP': map_score,
        'MRR': mrr_score,
        'Precision': precision_list,
        'Recall': recall_list
    }
    return evaluation_metrics

# إعداد البيانات وإنشاء مصفوفة TF-IDF
collection = "collection"  # اسم المجموعة الخاصة بك

file_path = f"C:/Users/lenovo/Desktop/ir-pro/lotte/lifestyle/dev/{collection}.tsv"
processed_texts = prepare_data(file_path)

tfidf_matrix, vectorizer = create_tfidf_matrix(processed_texts)

# حفظ الـ TF-IDF matrix و vectorizer في ملفات
tfidf_matrix_file = "C:/Users/lenovo/Desktop/ir-pro/lotte/lifestyle/dev/tfidf_matrix.pkl"
vectorizer_file = "C:/Users/lenovo/Desktop/ir-pro/lotte/lifestyle/dev/vectorizer.pkl"
with open(tfidf_matrix_file, 'wb') as fout:
    pickle.dump(tfidf_matrix, fout)
with open(vectorizer_file, 'wb') as fout:
    pickle.dump(vectorizer, fout)

# تحميل الـ TF-IDF matrix و vectorizer من الملفات باستخدام load
with open(tfidf_matrix_file, 'rb') as fin:
    tfidf_matrix = pickle.load(fin)
with open(vectorizer_file, 'rb') as fin:
    vectorizer = pickle.load(fin)

# حساب مقاييس التقييم
evaluation_file_path = r"C:/Users/lenovo/Desktop/ir-pro/lotte/lifestyle/dev/qas.search.jsonl"
evaluation_metrics = calculate_evaluation_metrics(evaluation_file_path, vectorizer, tfidf_matrix)

# طباعة متوسط الدقة (MAP) ومعدل الترتيب العكسي (MRR)
print(f"Mean Average Precision (MAP): {evaluation_metrics['MAP']}")
print(f"Mean Reciprocal Rank (MRR): {evaluation_metrics['MRR']}")

# طباعة الدقة والاستدعاء لكل الاستعلامات
for i, (precision, recall) in enumerate(zip(evaluation_metrics['Precision'], evaluation_metrics['Recall'])):
    print(f"Query {i+1}: Precision = {precision}, Recall = {recall}")