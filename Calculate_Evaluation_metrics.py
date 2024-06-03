import json
from collections import defaultdict

from Search_Index import Searcher


class Evaluator:
    evaluation_file_path = r"D:/iR/antique/output.jsonl"

    def __init__(self, evaluation_file_path, vectorizer, tfidf_matrix):
        self.evaluation_file_path = evaluation_file_path
        self.tfidf_matrix = tfidf_matrix
        self.vectorizer = vectorizer
        self.searcher = Searcher(vectorizer, tfidf_matrix)

    def calculate_evaluation_metrics(self):
        with open(self.evaluation_file_path, 'r', encoding='utf-8') as evaluation_file:
            data = [json.loads(line.strip()) for line in evaluation_file]
        relevant_docs = defaultdict(list)
        for item in data:
            relevant_docs[item['qid']].extend(item['answer_pids'])

        precision_list = []
        recall_list = []
        average_precision_sum = 0
        reciprocal_rank_sum = 0

        for query_data in data:
            query_text = query_data['query']
            relevant_pids = query_data['answer_pids']

            search_results = self.searcher.search_index(query_text, self.tfidf_matrix)
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
