from sklearn.feature_extraction.text import TfidfVectorizer

class TFIDFService:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def create_tfidf_matrix(self, processed_texts):
        tfidf_matrix = self.vectorizer.fit_transform(processed_texts)
        print(tfidf_matrix)
        return tfidf_matrix, self.vectorizer