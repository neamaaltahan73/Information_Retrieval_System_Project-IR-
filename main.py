import pickle
from Calculate_Evaluation_metrics import Evaluator
from Fetch_Data import PrepareData
from Tfidf_Matrix import TFIDFService

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from Search_Index import Searcher 
from Query_Refinement import handle_query
import LifeStyle_Offline 
import Antique_Offline

with open(LifeStyle_Offline.tfidf_matrix_file, 'rb') as fin:
    LifeStyle_tfidf_matrix = pickle.load(fin)
with open(LifeStyle_Offline.vectorizer_file, 'rb') as fin:
    LifeStyle_vectorizer = pickle.load(fin)

with open(Antique_Offline.tfidf_matrix_file, 'rb') as fin:
    Antique_tfidf_matrix = pickle.load(fin)
with open(Antique_Offline.vectorizer_file, 'rb') as fin:
    Antique_vectorizer = pickle.load(fin)

searcher = Searcher(LifeStyle_vectorizer, LifeStyle_tfidf_matrix, top_n=10)

evaluation_file_path = r"C:/Users/lenovo/Desktop/ir-pro/lotte/lifestyle/dev/qas.search.jsonl"
evaluation_file_path = r"C:/Users/lenovo/Desktop/ir-pro/anique/qas.search.jsonl"

evaluator = Evaluator(evaluation_file_path, LifeStyle_vectorizer, LifeStyle_tfidf_matrix)
evaluation_metrics = evaluator.calculate_evaluation_metrics()

print(f"Mean Average Precision (MAP): {evaluation_metrics['MAP']}")
print(f"Mean Reciprocal Rank (MRR): {evaluation_metrics['MRR']}")

for i, (precision, recall) in enumerate(zip(evaluation_metrics['Precision'], evaluation_metrics['Recall'])):
    print(f"Query {i+1}: Precision = {precision}, Recall = {recall}")

# ////////////////////////////////////////////////////////////////////////////////////////////////////// Query Refinement


query = ""

# interface_search_index = searcher.interface_search_index(query, tfidf_matrix)

# Function to handle the search button click
def on_search():
    query = query_entry.get()
    dataset = dataset_var.get()
    if dataset == "Dataset 1":
        searcher = Searcher(LifeStyle_vectorizer, LifeStyle_tfidf_matrix)
        with open(LifeStyle_Offline.output_file, 'r') as file:
             content = file.read()
        documents = content.split('\n')
        search_suggestions = searcher.interface_search_index(query, documents)
        result = "\n\n".join([f"Document {index}:\n{documents}" for index, documents in search_suggestions])    
    if dataset == "Dataset 2": 
        searcher = Searcher(Antique_vectorizer, Antique_tfidf_matrix)
        with open(Antique_Offline.output_file, 'r') as file:
             content = file.read()
        documents = content.split('\n')
        search_suggestions = searcher.interface_search_index(query, documents)
        result = "\n\n".join([f"Document {index}:\n{documents}" for index, documents in search_suggestions]) 
    if dataset not in ["Dataset 1", "Dataset 2"]:
        result = "Please select a dataset."       
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, result)


# Function to handle the complete button click
def on_complete():
    query = query_entry.get()
    dataset = dataset_var.get()
    if dataset == "Dataset 1":
        suggestions = handle_query(query, LifeStyle_Offline.file_path)
        result = "\n\n\n".join(suggestions)
    if dataset == "Dataset 2": 
        suggestions = handle_query(query, Antique_Offline.file_path)
        result = "\n\n\n".join(suggestions)
    if dataset not in ["Dataset 1", "Dataset 2"]:
        result = "Please select a dataset."     
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, result)


# Create the main window
root = tk.Tk()
root.title("Information Retrieval System")

# Set the window size
window_width = 800
window_height = 600
root.geometry(f"{window_width}x{window_height}")

# Center the window on the screen
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
position_top = int(screen_height / 2 - window_height / 2)
position_right = int(screen_width / 2 - window_width / 2)
root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

# Frame for query input and complete button
query_frame = tk.Frame(root)
query_frame.pack(pady=5)

# Query input
query_label = tk.Label(query_frame, text="Enter your query:")
query_label.pack(side=tk.LEFT, padx=5)
query_entry = tk.Entry(query_frame, width=80)
query_entry.pack(side=tk.LEFT, padx=5)

# Complete button
complete_button = tk.Button(query_frame, text="Complete", command=on_complete)
complete_button.pack(side=tk.LEFT, padx=5)

# Dataset selection
dataset_var = tk.StringVar()
dataset_label = tk.Label(root, text="Choose a dataset:")
dataset_label.pack(pady=5)
dataset_option1 = ttk.Radiobutton(root, text="Lifestyle", variable=dataset_var, value="Dataset 1")
dataset_option1.pack(pady=2)
dataset_option2 = ttk.Radiobutton(root, text="Anitque", variable=dataset_var, value="Dataset 2")
dataset_option2.pack(pady=2)

# Search button
search_button = tk.Button(root, text="Search", command=on_search)
search_button.pack(pady=10)

# Result display
result_label = tk.Label(root, text="Results:")
result_label.pack(pady=5)
result_text = ScrolledText(root, wrap=tk.WORD, width=90, height=20)
result_text.pack(pady=10)

# Start the GUI event loop
root.mainloop()