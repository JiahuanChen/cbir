import cv2
import numpy
import os
import glob
try:
    import hdidx
except:
    import hdidx

from cbir.database.memory import Database
from cbir.kmeans.sklearn import KMeans
from sklearn import preprocessing
from cbir.utils import dump,load
from sklearn.decomposition import PCA

sift = cv2.xfeatures2d.SIFT_create()

# Extract features from an image
def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    kp, des = sift.detectAndCompute(image, None)
    if des is None:
        des = numpy.empty([0, 128], dtype='float32')
    return des

# [1, 1, 1, 2, 3, 3] => { 1: 3, 2: 1, 3: 2 }
def occurences(words):
    unique, counts = numpy.unique(words, return_counts=True)
    return dict(zip(unique, counts))


class BoWEngine():
    def __init__(self, data_directory='./data'):
        self.data_directory = data_directory
        self.kmeans = KMeans(
            dictionary_path=self.datapath('dictionary.pickle')
        )
        self.database = Database(
            database_path=self.datapath('database.pickle')
        )

    def datapath(self, filename):
        return os.path.join(self.data_directory, filename)

    def __assert_dictionary_exists(self):
        if not self.kmeans.dictionary_exists():
            raise Exception('Need to load or train dictionary')

    # Builds BoW dictionary from list of features
    def train(self, clusters=1000, limit=None, path='./images/**'):
        self.kmeans = KMeans(
            dictionary_size=clusters,
            dictionary_path=self.datapath('dictionary.pickle')
            )
        all_features = numpy.empty([0, 128], dtype='float32')
        i = 0
        for filepath in glob.iglob(os.path.join(path, '*.jpg')):
            i += 1
            if (i > limit):
                break
            features = extract_features(filepath)
            all_features = numpy.vstack((all_features, features))
            # print(all_features.shape)

        if len(all_features) == 0:
            raise Exception('No feature extracted from "%s"' % path)

        centroids  = self.kmeans.fit(all_features)
        self.kmeans.save(self.datapath('dictionary.pickle'))

        return centroids

    def _build_bow(self, image):
        self.__assert_dictionary_exists()
        features = extract_features(image)
        words = self.kmeans.predict(features)
        histo = occurences(words)
        return histo

    def get_bow(self, image):
        return self._build_bow(image)

    def get_bow_vector(self, image):
        bow = self._build_bow(image)
        v = []
        for i in range(self.kmeans.dictionary_size):
            value = bow.get(i, 0)
            v.append(value)
        return v

    # reduction and indexation; save index imformation
    def rdcidx(self, subvectors = 4):
        vlad = load('vlad.pickle')
        #pca
        pca = PCA(n_components=128)
        vlad = pca.fit_transform(vlad)
        #adc
        idx = hdidx.indexer.PQIndexer()
        idx.set_storage('lmdb', {'path': './pq.idx'})
        idx.build({'vals': vlad, 'nsubq': subvectors})
        idx.add(vlad)
        idx.save('./pq.info')

    # Indexes an image through the database
    def index(self, image_path):
        bow = self._build_bow(image_path)

        self.database.insert(
            bow,
            image_path=image_path,
        )
        return bow

    def delete_index(self, index_name=None):
        return self.database.delete_index(index_name=index_name)

    # Queries an image
    def query(self, image_path):
        bow = self._build_bow(image_path)
        # print(bow)
        return self.database.query(bow)

    # Returns number of indexed images
    def count(self):
        return self.database.count()

    # calculate vlad for each image
    def vlad(self, image_path):
        # get image descreptors
        des = extract_features(image_path)
        # get dictionary
        cen = load("./data/dictionary.pickle")
        # calculate the vector of locally aggregated descriptors
        VLAD = numpy.zeros(cen.cluster_centers_.shape)
        des_size = len(des)
        for j in range(des_size):
            #pridect the NN(x)
            nnx = self.kmeans.predict(des[0].reshape(1,-1))[0]
            VLAD[nnx] += des[j] - cen.cluster_centers_[nnx]
        #l2 mornalization
        VLAD_L2_NORM = preprocessing.normalize(VLAD, norm='l2')
        #convert the 16*128 matrix to 1*2048
        VLAD_L2_NORM = VLAD_L2_NORM.reshape(1,-1)
        return VLAD_L2_NORM

    #iterate the selected images
    def cal_vlad(self, limit, path):
        i = 0
        all_vlad = numpy.empty([0, 128*16], dtype='float32')
        # iterate the images
        for filepath in glob.iglob(os.path.join(path, '*.jpg')):
            i += 1
            if (i > limit):
                break
            print('calculating %s' % filepath)
            VLAD = self.vlad(filepath)
            all_vlad = numpy.vstack((all_vlad, VLAD[0]))
        # save the (limite*16) by (128) vlad
        dump(all_vlad,"vlad.pickle")

    #find the 5nn
    def query_vlad(self, path):
        #get vlad
        vlad = self.vlad(path)
        vlad_dict = load('vlad.pickle')
        #PCA
        pca = PCA(n_components=128)
        pca.fit(vlad_dict)
        vlad = pca.transform(vlad)
        # laod ADC
        idx = hdidx.indexer.PQIndexer()
        idx.set_storage('lmdb', {'path': './pq.idx'})
        idx.load('./pq.info')
        return idx.search(vlad,5)
