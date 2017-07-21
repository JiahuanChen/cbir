import sklearn.cluster as skcluster
from cbir.utils import load, dump


class KMeans():
    def __init__(
        self,
        dictionary_size=10,
        dictionary_path=None,
    ):
        self.dictionary_path = dictionary_path
        # http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        self.kmeans = skcluster.KMeans(
            n_clusters=dictionary_size,
            max_iter=300,
            n_jobs=1,
        )
        self.load()

    def dictionary_exists(self):
        return self.dictionary is not None

    def load(self):
        self.dictionary = load(self.dictionary_path)

    def save(self, out=False):
        if not out:
            out = self.dictionary_path
        dump(self.dictionary, out)

    # trains dictionary...
    def fit(self, descs):
        print('fitting')
        res = self.kmeans.fit(descs)
        print(res)
        self.dictionary = self.kmeans
        return res

    def predict(self, x):
        return self.dictionary.predict(x)
