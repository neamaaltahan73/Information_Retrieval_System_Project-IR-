from sklearn.metrics.pairwise import cosine_similarity


class SimilarityCalculator:
    def compute_similarity(self, query_vector, tfidf_matrix):
        cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)
        return cosine_similarities
