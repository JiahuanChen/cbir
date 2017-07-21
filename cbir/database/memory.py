# In memory implementation of inverted index
from cbir.utils import load, dump, delete


def sparse_vector_dot(a, b):
    result = 0
    for vector_column, value in a.items():
        if vector_column in b:
            result += b[vector_column] * value
    return result


def l2_norm(a):
    norm = 0
    for key, value in a.items():
        norm += value ** 2
    return norm


def cos_similarity(a, b):
    return sparse_vector_dot(a, b) / ((l2_norm(a) * l2_norm(b)) ** (1/2))


class Database():
    def __init__(self, database_path=None, min_score=0.1):
        self.index_created = False
        self.database_path = database_path
        self.min_score = min_score

        self.documents = {}
        self.postings_list = {}
        self.load_from_disk()

    def load_from_disk(self):
        data = load(self.database_path)
        if (data):
            self.documents = data['documents']
            self.postings_list = data['postings']

    def save_to_disk(self):
        data = {'postings': self.postings_list, 'documents': self.documents}
        dump(data, self.database_path)

    def delete_index(self, index_name=None):
        delete(self.database_path)

    def insert(self, bow, image_path=None):
        image_id = image_path
        for word in bow:
            image_ids = self.postings_list.get(word, set())
            image_ids.add(image_id)
            self.postings_list[word] = image_ids
        self.documents[image_id] = bow

    def query(self, query_bow):
        candidate_ids = set()
        for word in query_bow:
            candidate_ids = candidate_ids.union(
                self.postings_list.get(word, set())
            )

        candidates = []
        for image_id in list(candidate_ids):
            candidate_bow = self.documents[image_id]
            score = cos_similarity(query_bow, candidate_bow)
            candidates.append({'image_id': image_id, 'score': score})

        candidates = sorted(
            candidates,
            key=lambda x: x['score'],
            reverse=True,
        )

        # todo: filter out scores lower than min_score

        return candidates
