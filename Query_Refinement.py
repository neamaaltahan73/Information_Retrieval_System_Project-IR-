from Fetch_Data import PrepareData
from fast_autocomplete import AutoComplete
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
from nltk.corpus import wordnet
from spellchecker import SpellChecker 
import glob
from fast_autocomplete import AutoComplete


# Load NLTK and Spacy libraries
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")


# الطلب الاضافي query refinement
def complete(query, file_path):
    collection = "collection"
    prepare = PrepareData(collection)
    queries = prepare.prepare_data(file_path)
    words = {}
    for value in queries:
        value = value.strip()
        if value.startswith(query):
            new_key_values_dict = {value: {}}
            words.update(new_key_values_dict)
    autocomplete = AutoComplete(words=words)
    suggestions = autocomplete.search(query, max_cost=10, size=10)
    return suggestions

def suggest_spelling_corrections(query):
    tokens = nltk.word_tokenize(query)
    spellchecker = SpellChecker()
    corrections = []
    for token in tokens:
        if token.lower() not in spellchecker:
            correction = spellchecker.correction(token)
            corrections.append(correction)
    return corrections

def expand_query(query):
    synonyms = set()
    doc = nlp(query)
    for token in doc:
        for syn in wordnet.synsets(token.text):
            for lemma in syn.lemmas():
                if lemma.name() != token.text:
                    synonyms.add(lemma.name())
    expanded_query = query.split() + list(synonyms)
    return expanded_query

def handle_query(query, file_path):
    autocomplete_suggestions = complete(query, file_path)
    autocomplete_suggestions_list = [f"Autocomplete suggestion: {suggestion}" for suggestion in autocomplete_suggestions]
    spelling_corrections = suggest_spelling_corrections(query)
    spelling_corrections_list = [f"Spelling correction: {correction}" for correction in spelling_corrections]
    expanded_query = expand_query(query)
    expanded_query_list = [f"Expanded query: {expanded}" for expanded in expanded_query]
    all_suggestions = autocomplete_suggestions_list + spelling_corrections_list + expanded_query_list
    return all_suggestions
