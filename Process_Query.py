from Clean_Text import CleanText

class QueryProcessor:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        self.cleaner = CleanText()

    def process_query(self, query):
        processed_query = self.cleaner.clean_text(query)
        query_vector = self.vectorizer.transform([processed_query])
        print(query_vector)
        return query_vector
