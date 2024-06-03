import pickle
from Calculate_Evaluation_metrics import Evaluator
from Fetch_Data import PrepareData
from Tfidf_Matrix import TFIDFService

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from Search_Index import Searcher 
from Query_Refinement import handle_query
import os

collection = "collection"
output_file = "output_file"

prepare = PrepareData(collection)

file_path = f"C:/Users/lenovo/Desktop/ir-pro/antique/{collection}.tsv"
output_file = f"C:/Users/lenovo/Desktop/ir-pro/antique/{output_file}.tsv"
tfidf_matrix_file = "C:/Users/lenovo/Desktop/ir-pro/antique/tfidf_matrix.pkl"
vectorizer_file = "C:/Users/lenovo/Desktop/ir-pro/antique/vectorizer.pkl"

data_preparer = prepare.prepare_data(file_path)

# Save the prepared data to the output file
if data_preparer is not None:
    # Check if data_preparer is a list and join it into a string
    if isinstance(data_preparer, list):
        data_preparer = "\n".join(data_preparer)
    
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(data_preparer)
else:
    raise ValueError("Prepared data is None. Please ensure the prepare_data method is working correctly.")

tfidf_service = TFIDFService()

# Read the content of the output file
with open(output_file, 'r') as file:
    content = file.read()

# Split the content into a list of documents
documents = content.split('\n')

# Create TF-IDF matrix using the content of the output file
tfidf_matrix, vectorizer = tfidf_service.create_tfidf_matrix(documents)


with open(tfidf_matrix_file, 'wb') as fout:
    pickle.dump(tfidf_matrix, fout)
with open(vectorizer_file, 'wb') as fout:
    pickle.dump(vectorizer, fout)
