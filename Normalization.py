
# import string
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# import numpy as np
# import re
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer, WordNetLemmatizer
# import contractions
# import spacy
# import csv
# import nltk
# import pickle
# from collections import defaultdict
# import os
# nlp = spacy.load("en_core_web_sm")
# import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.data.path.append("D:/iR/nltk_data")
# from sklearn.feature_selection import SelectKBest, f_classif
# # تحميل مكتبات NLTK و Spacy
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('stopwords')

# nlp = spacy.load("en_core_web_sm")
# from sklearn.model_selection import train_test_split
# # تابع لتحضير البيانات باستخدام TF-IDF
# def prepare_tfidf_matrix(texts):
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform(texts)
#     feature_names = vectorizer.get_feature_names_out()
#     return tfidf_matrix, feature_names, vectorizer

# # تابع لتحضير بيانات مجموعة البيانات
# def prepare_data_for_tfidf(file_path, num_lines=500):
#     lines_after_read = []
#     with open(file_path, "rt", encoding="utf8") as fin:
#         for i, line in enumerate(fin):
#             if i < num_lines:
#                 processed_line = execute_operations_on_text(line.strip())
#                 lines_after_read.append(processed_line)
#             else:
#                 break  # تحقق من عدد الأسطر المقروءة
#     return lines_after_read

# # تابع لتحويل النص إلى تنسيق مناسب
# def execute_operations_on_text(text):
#     tokenized = Tokenization_line(text)
#     lowercased = lowercase_and_remove_punctuation(tokenized)
#     no_stopwords = remove_stopwords(lowercased)
#     no_numbers = remove_numbers(no_stopwords)
#     lemmatized_stemmed = lemmatize_and_stem_line(no_numbers)
#     stemmed = stemming_line(lemmatized_stemmed)
#     no_whitespace = remove_white_space(stemmed)
#     return no_whitespace

# # توابع معالجة النصوص
# def Tokenization_line(line):
#     word_tokens = word_tokenize(line.strip())
#     return word_tokens

# def lowercase_and_remove_punctuation(word_tokens):
#     lowercase_line = ' '.join(word_tokens).lower()
#     no_punctuation_line = lowercase_line.translate(str.maketrans('', '', string.punctuation))
#     return no_punctuation_line

# def remove_stopwords(text):
#     doc = nlp(text)
#     words = [token.text for token in doc if not token.is_stop]
#     filtered_text = ' '.join(words)
#     return filtered_text

# def remove_numbers(filtered_line):
#     no_numbers_line = re.sub(r'\d+', '', filtered_line)
#     return no_numbers_line

# def lemmatize_and_stem_line(no_numbers_line):
#     lemmatizer = WordNetLemmatizer()
#     stemmer = PorterStemmer()
#     words = word_tokenize(no_numbers_line)
#     lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in words]
#     stemmed_words = [stemmer.stem(word) for word in lemmatized_words]
#     lemmatized_stemmed_line = ' '.join(stemmed_words)
#     return lemmatized_stemmed_line

# def stemming_line(lemmatized_stemmed_line):
#     stemmer = PorterStemmer()
#     words = word_tokenize(lemmatized_stemmed_line.strip())
#     stemmed_words = [stemmer.stem(word) for word in words]
#     stemmed_line = ' '.join(stemmed_words)
#     return stemmed_line

# def remove_white_space(line):
#     line = re.sub(r'\s+', ' ', line)
#     return line

# # تابع لإنشاء قاموس للاختصارات
# def create_acronym_dic(acronymFileName):
#     acronym = {}
#     try:
#         with open(acronymFileName, encoding="utf8") as file:
#             csv_file = csv.reader(file, delimiter="\t")
#             for line in csv_file:
#                 acronym[line[0]] = line[1]
#         print("Acronym dictionary created successfully.")
#     except Exception as e:
#         print(f"Error reading acronym file: {e}")
#     return acronym

# # تابع لاستبدال الاختصارات في النصوص
# def replace_acronym(text, mydict):
#     words = text.split()
#     replaced_text = ' '.join([mydict.get(word, word) for word in words])
#     return replaced_text
# def tfidf_representation(collection, acronymFileName):
#     file_path = f"D:/iR/lotte/lifestyle/dev/{collection}.tsv"
#     output_file_path = f"D:/iR/lotte/lifestyle/dev/expended_.tsv"
    
#     # إنشاء قاموس الاختصارات
#     acronym_dic = create_acronym_dic(acronymFileName)
#     print("Acronym Dictionary:", acronym_dic)  # طباعة القاموس للتحقق

#     processed_texts = prepare_data_for_tfidf(file_path)
    
#     # استبدال الاختصارات في النصوص
#     processed_texts = [replace_acronym(text, acronym_dic) for text in processed_texts]
#     print("Processed Texts after acronym replacement:", processed_texts[:10])  # طباعة النصوص بعد استبدال الاختصارات
    
#     tfidf_matrix, feature_names, vectorizer = prepare_tfidf_matrix(processed_texts)
    
#     print("TF-IDF Matrix for the first 10 documents:")
#     print(tfidf_matrix.toarray()[:10])  # لعرض مصفوفة TF-IDF
    
#     # كتابة النصوص المعالجة في ملف جديد
#     with open(output_file_path, "w", encoding="utf8") as fout:
#         for text in processed_texts:
#             fout.write(text + "\n")
    
#     # طباعة الميزات وأوزانها لكل وثيقة
#     for i, text in enumerate(processed_texts[:10]):
#         print(f"\nDocument {i + 1}:")
#         feature_index = tfidf_matrix[i,:].nonzero()[1]
#         tfidf_scores = zip(feature_index, [tfidf_matrix[i, x] for x in feature_index])
#         for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
#             print(f"{w}: {s}")
    
#     return tfidf_matrix, feature_names, vectorizer, processed_texts

# # اختبار التابع
# collection = "collection"  # اسم المجموعة الخاصة بك
# acronymFileName = "D:/iR/lotte/lifestyle/dev/acrony.csv"
# tfidf_matrix, feature_names, vectorizer, processed_texts = tfidf_representation(collection, acronymFileName)


# # تابع لتنقية الميزات بناءً على عتبة معينة
# def filter_features_by_tfidf_threshold(tfidf_matrix, feature_names, threshold=0.1):
#     mask = np.asarray(tfidf_matrix.max(axis=0)).ravel() > threshold
#     filtered_tfidf_matrix = tfidf_matrix[:, mask]
#     filtered_feature_names = np.array(feature_names)[mask]
#     return filtered_tfidf_matrix, filtered_feature_names

# # تابع لتحضير البيانات باستخدام TF-IDF مع تنقية الميزات
# def prepare_tfidf_matrix_with_feature_selection(texts, threshold=0.1):
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform(texts)
#     feature_names = vectorizer.get_feature_names_out()
    
#     filtered_tfidf_matrix, filtered_feature_names = filter_features_by_tfidf_threshold(tfidf_matrix, feature_names, threshold)
    
#     return filtered_tfidf_matrix, filtered_feature_names, vectorizer

# def tfidf_representation(collection, acronymFileName, threshold=0.1):
#     file_path = f"D:/iR/lotte/lifestyle/dev/{collection}.tsv"
#     output_file_path = f"D:/iR/lotte/lifestyle/dev/expended_.tsv"
    
#     # إنشاء قاموس الاختصارات
#     acronym_dic = create_acronym_dic(acronymFileName)
#     print("Acronym Dictionary:", acronym_dic)  # طباعة القاموس للتحقق

#     processed_texts = prepare_data_for_tfidf(file_path)
    
#     # استبدال الاختصارات في النصوص
#     processed_texts = [replace_acronym(text, acronym_dic) for text in processed_texts]
#     print("Processed Texts after acronym replacement:", processed_texts[:10])  # طباعة النصوص بعد استبدال الاختصارات
    
#     tfidf_matrix, feature_names, vectorizer = prepare_tfidf_matrix_with_feature_selection(processed_texts, threshold)
    
#     print("Filtered TF-IDF Matrix for the first 10 documents:")
#     print(tfidf_matrix.toarray()[:10])  # لعرض مصفوفة TF-IDF بعد التنقية
    
#     # كتابة النصوص المعالجة في ملف جديد
#     with open(output_file_path, "w", encoding="utf8") as fout:
#         for text in processed_texts:
#             fout.write(text + "\n")
    
#     # طباعة الميزات وأوزانها لكل وثيقة بعد التنقية
#     for i, text in enumerate(processed_texts[:10]):
#         print(f"\nDocument {i + 1}:")
#         feature_index = tfidf_matrix[i,:].nonzero()[1]
#         tfidf_scores = zip(feature_index, [tfidf_matrix[i, x] for x in feature_index])
#         for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
#             print(f"{w}: {s}")
    
#     return tfidf_matrix, feature_names, vectorizer, processed_texts

# # اختبار التابع
# collection = "collection"  # اسم المجموعة الخاصة بك
# acronymFileName = "D:/iR/lotte/lifestyle/dev/acrony.csv"
# tfidf_matrix, feature_names, vectorizer, processed_texts = tfidf_representation(collection, acronymFileName, threshold=0.1)



# # def build_index(tfidf_matrix, feature_names):
# #     index = defaultdict(list)
# #     for doc_id, row in enumerate(tfidf_matrix):
# #         for term_id in row.nonzero()[1]:
# #             term = feature_names[term_id]
# #             index[term].append((doc_id, row[0, term_id]))
# #     return index

# # def save_index(index, file_path):
# #     with open(file_path, 'wb') as fout:
# #         pickle.dump(index, fout)

# # def load_index(file_path):
# #     with open(file_path, 'rb') as fin:
# #         index = pickle.load(fin)
# #     return index

# # def process_query(query):
# #     query = execute_operations_on_text(query)
# #     return query

# # # ... جزء الكود السابق ...

# # def search_index(query, index, vectorizer, processed_texts):
# #     query = process_query(query)
# #     query_vector = vectorizer.transform([query])
# #     results = defaultdict(float)
    
# #     for term_id in query_vector.nonzero()[1]:
# #         term = vectorizer.get_feature_names_out()[term_id]
# #         if term in index:
# #             for doc_id, weight in index[term]:
# #                 results[doc_id] += weight * query_vector[0, term_id]
    
# #     sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
# #     return [(doc_id, score, processed_texts[doc_id]) for doc_id, score in sorted_results]

# # # بناء الفهرس
# # index = build_index(tfidf_matrix, feature_names)

# # # حفظ الفهرس في ملف
# # index_file_path = "D:/IR/lotte/lifestyle/dev/index.pkl"
# # save_index(index, index_file_path)

# # # تحميل الفهرس من الملف
# # loaded_index = load_index(index_file_path)

# # # البحث في الفهرس
# # query = "are deaf cats more aggressive?"
# # search_results = search_index(query, loaded_index, vectorizer, processed_texts)

# # # عرض النتائج
# # for doc_id, score, text in search_results[:10]:  # عرض أفضل 10 نتائج
# #     print(f"Document ID: {doc_id}, Score: {score}")
# #     print(f"Text: {text}")
# #     print("\n")




import json
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import contractions
import spacy
import csv
import nltk
import pickle
from collections import defaultdict

# Set the nltk data path
nltk.data.path.append("D:/iR/nltk_data")

# Load necessary NLTK and Spacy libraries
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

# Function to prepare TF-IDF matrix
def prepare_tfidf_matrix(texts):
    vectorizer = TfidfVectorizer(max_df=0.85, min_df=0.05, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names, vectorizer

# Function to prepare data for TF-IDF
def prepare_data_for_tfidf(file_path, num_lines=5000):
    lines_after_read = []
    with open(file_path, "rt", encoding="utf8") as fin:
        for i, line in enumerate(fin):
            if i < num_lines:
                processed_line = execute_operations_on_text(line.strip())
                lines_after_read.append(processed_line)
            else:
                break
    return lines_after_read

# Function to execute text preprocessing operations
def execute_operations_on_text(text):
    tokenized = Tokenization_line(text)
    lowercased = lowercase_and_remove_punctuation(tokenized)
    no_stopwords = remove_stopwords(lowercased)
    no_numbers = remove_numbers(no_stopwords)
    lemmatized_stemmed = lemmatize_and_stem_line(no_numbers)
    stemmed = stemming_line(lemmatized_stemmed)
    no_whitespace = remove_white_space(stemmed)
    return no_whitespace

# Text processing functions
def Tokenization_line(line):
    word_tokens = word_tokenize(line.strip())
    return word_tokens

def lowercase_and_remove_punctuation(word_tokens):
    lowercase_line = ' '.join(word_tokens).lower()
    no_punctuation_line = lowercase_line.translate(str.maketrans('', '', string.punctuation))
    return no_punctuation_line

def remove_stopwords(text):
    doc = nlp(text)
    words = [token.text for token in doc if not token.is_stop]
    filtered_text = ' '.join(words)
    return filtered_text

def remove_numbers(filtered_line):
    no_numbers_line = re.sub(r'\d+', '', filtered_line)
    return no_numbers_line

def lemmatize_and_stem_line(no_numbers_line):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    words = word_tokenize(no_numbers_line)
    lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in words]
    stemmed_words = [stemmer.stem(word) for word in lemmatized_words]
    lemmatized_stemmed_line = ' '.join(stemmed_words)
    return lemmatized_stemmed_line

def stemming_line(lemmatized_stemmed_line):
    stemmer = PorterStemmer()
    words = word_tokenize(lemmatized_stemmed_line.strip())
    stemmed_words = [stemmer.stem(word) for word in words]
    stemmed_line = ' '.join(stemmed_words)
    return stemmed_line

def remove_white_space(line):
    line = re.sub(r'\s+', ' ', line)
    return line

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

# Function to represent texts using TF-IDF
def tfidf_representation(collection, acronymFileName):
    file_path = f"C:/Users/lenovo/Desktop/ir-pro/lotte/lifestyle/dev/{collection}.tsv"
    output_file_path = f"C:/Users/lenovo/Desktop/ir-pro/lotte/lifestyle/dev/expended_.tsv"
    
    # Create acronym dictionary
    acronym_dic = create_acronym_dic(acronymFileName)
    print("Acronym Dictionary:", acronym_dic)

    processed_texts = prepare_data_for_tfidf(file_path)
    
    # Replace acronyms in texts
    processed_texts = [replace_acronym(text, acronym_dic) for text in processed_texts]
    print("Processed Texts after acronym replacement:", processed_texts[:10])
    
    tfidf_matrix, feature_names, vectorizer = prepare_tfidf_matrix(processed_texts)
    
    print("TF-IDF Matrix for the first 10 documents:")
    print(tfidf_matrix.toarray()[:10])
    
    # Write processed texts to a new file
    with open(output_file_path, "w", encoding="utf8") as fout:
        for text in processed_texts:
            fout.write(text + "\n")
    
    # Print features and their weights for each document
    for i, text in enumerate(processed_texts[:10]):
        print(f"\nDocument {i + 1}:")
        feature_index = tfidf_matrix[i,:].nonzero()[1]
        tfidf_scores = zip(feature_index, [tfidf_matrix[i, x] for x in feature_index])
        for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
            print(f"{w}: {s}")
    
    return tfidf_matrix, feature_names, vectorizer, processed_texts

# Test the function
collection = "collection"
acronymFileName = "C:/Users/lenovo/Desktop/ir-pro/lotte/lifestyle/dev/acrony.csv"
tfidf_matrix, feature_names, vectorizer, processed_texts = tfidf_representation(collection, acronymFileName)


def build_index(tfidf_matrix, feature_names, threshold=0.1):
    index = defaultdict(list)
    for doc_id, row in enumerate(tfidf_matrix):
        for term_id in row.nonzero()[1]:
            term = feature_names[term_id]
            if row[0, term_id] >= threshold:  # فهرس فقط الكلمات التي تتجاوز الحد الأدنى
                index[term].append((doc_id, row[0, term_id]))
    return index

def save_index(index, file_path):
    with open(file_path, 'wb') as fout:
        pickle.dump(index, fout)

def load_index(file_path):
    with open(file_path, 'rb') as fin:
        index = pickle.load(fin)
    return index

def process_query(query):
    query = execute_operations_on_text(query)
    return query

def search_index(query, index, vectorizer, processed_texts):
    query = process_query(query)
    query_vector = vectorizer.transform([query])
    results = defaultdict(float)
    
    for term_id in query_vector.nonzero()[1]:
        term = vectorizer.get_feature_names_out()[term_id]
        if term in index:
            for doc_id, weight in index[term]:
                results[doc_id] += weight * query_vector[0, term_id]
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    return [(doc_id, score, processed_texts[doc_id]) for doc_id, score in sorted_results]

# بناء الفهرس مع حد أدنى لقيمة TF-IDF
index = build_index(tfidf_matrix, feature_names, threshold=0.1)

# حفظ الفهرس في ملف
index_file_path = "C:/Users/lenovo/Desktop/ir-pro/lotte/lifestyle/dev/index.pkl"
save_index(index, index_file_path)

# تحميل الفهرس من الملف
loaded_index = load_index(index_file_path)

# البحث في الفهرس
query = "how much should i feed my 1 year old english mastiff?"
search_results = search_index(query, loaded_index, vectorizer, processed_texts)

# عرض النتائج
for doc_id, score, text in search_results[:10]:  # عرض أفضل 10 نتائج
    print(f"Document ID: {doc_id}, Score: {score}")
    print(f"Text: {text}")
    print("\n")

# حساب مقاييس التقييم
def calculate_evaluation_metrics(evaluation_file_path):
    # قراءة ملف التقييم
    with open(evaluation_file_path, 'r') as evaluation_file:
        data = [json.loads(line.strip()) for line in evaluation_file]
        # عملية معالجة بيانات التقييم
    relevant_docs = defaultdict(list)
    for item in data:
        relevant_docs[item['qid']].extend(item['answer_pids'])

    # احتساب مقاييس التقييم
    precision_list = []
    recall_list = []
    average_precision_sum = 0
    reciprocal_rank_sum = 0

    for query_id, relevant_pids in relevant_docs.items():
        retrieved_pids = [result[0] for result in search_results]
        
        # حساب الدقة والاستدعاء
        relevant_retrieved = len(set(relevant_pids) & set(retrieved_pids))
        precision = relevant_retrieved / len(retrieved_pids) if len(retrieved_pids) > 0 else 0
        recall = relevant_retrieved / len(relevant_pids) if len(relevant_pids) > 0 else 0
        
        precision_list.append(precision)
        recall_list.append(recall)

        # حساب متوسط الدقة
        average_precision = 0
        relevant_count = 0
        for i, pid in enumerate(retrieved_pids):
            if pid in relevant_pids:
                relevant_count += 1
                average_precision += relevant_count / (i + 1)
        average_precision /= len(relevant_pids)
        average_precision_sum += average_precision

        # حساب المرتبة المتبادلة المتوسطة
        for i, pid in enumerate(retrieved_pids):
            if pid in relevant_pids:
                reciprocal_rank_sum += 1 / (i + 1)
                break 
             # نهاية الفهرس
        
    # حساب مقاييس التقييم
    query_count = len(relevant_docs)
    map_score = average_precision_sum / query_count if query_count > 0 else 0.0
    mrr_score = reciprocal_rank_sum / query_count if query_count > 0 else 0.0

    # إعادة مقاييس التقييم
    evaluation_metrics = {
        'MAP': map_score,
        'MRR': mrr_score,
        'Precision': precision_list,
        'Recall': recall_list
    }
    print(f"Mean Average Precision (MAP): {map_score}")
    return evaluation_metrics

# مثال على الاستخدام
evaluation_file_path = r"C:/Users/lenovo/Desktop/ir-pro/lotte/lifestyle/dev/qas.search.jsonl"
evaluation_metrics = calculate_evaluation_metrics(evaluation_file_path)

# طباعة الدقة والاستدعاء
for i, precision in enumerate(evaluation_metrics['Precision']):
    recall = evaluation_metrics['Recall'][i]
    print(f"Query {i+1}: Precision = {precision}, Recall = {recall}")

def calculate_map(evaluation_file_path):
    # قراءة ملف التقييم
    with open(evaluation_file_path, 'r') as evaluation_file:
        data = [json.loads(line.strip()) for line in evaluation_file]

    # عملية معالجة بيانات التقييم
    relevant_docs = defaultdict(list)
    for item in data:
        relevant_docs[item['qid']].extend(item['answer_pids'])

    # احتساب مقياس متوسط الدقة (MAP)
    average_precision_sum = 0

    for query_id, relevant_pids in relevant_docs.items():
        retrieved_pids = [result[0] for result in search_results]
        
        # حساب متوسط الدقة
        average_precision = 0
        relevant_count = 0
        for i, pid in enumerate(retrieved_pids):
            if pid in relevant_pids:
                relevant_count += 1
                average_precision += relevant_count / (i + 1)
        average_precision /= len(relevant_pids)
        average_precision_sum += average_precision

    # حساب قيمة متوسط الدقة (MAP)
    query_count = len(relevant_docs)
    map_score = average_precision_sum / query_count if query_count > 0 else 0.0
    
    return map_score

# استخدام الدالة لحساب متوسط الدقة (MAP)
map_score = calculate_map(evaluation_file_path)
print(f"Mean Average Precision (MAP): {map_score}")

#####################


