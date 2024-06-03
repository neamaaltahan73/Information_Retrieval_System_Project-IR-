from Compute_Similarity import SimilarityCalculator
from Process_Query import QueryProcessor
from Fetch_Data import PrepareData

class Searcher:

    def __init__(self, vectorizer, tfidf_matrix, top_n=10):
        self.query_processor = QueryProcessor(vectorizer)
        self.similarity_calculator = SimilarityCalculator()
        self.tfidf_matrix = tfidf_matrix
        self.top_n = top_n

    def search_index(self, query):
        query_vector = self.query_processor.process_query(query)
        cosine_similarities = self.similarity_calculator.compute_similarity(query_vector, self.tfidf_matrix)
        sorted_indices = cosine_similarities[0].argsort()[::-1][:self.top_n]  # احتفظ بأول top_n نتائج فقط
        results = [(index, cosine_similarities[0][index]) for index in sorted_indices]
        return results
    
    def interface_search_index(self, query, documents):
        query_vector = self.query_processor.process_query(query)
        cosine_similarities = self.similarity_calculator.compute_similarity(query_vector, self.tfidf_matrix)
        sorted_indices = cosine_similarities[0].argsort()[::-1][:self.top_n]  # احتفظ بأول top_n نتائج فقط
        query_output = [(index, documents[index]) for index in sorted_indices]
        return query_output
