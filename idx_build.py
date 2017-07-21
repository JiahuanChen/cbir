# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 23:34:00 2017

@author: blowv6
"""
import sys
sys.path.append('E:/binded/challenge/cbir-challenge-jiahuanchen')
import numpy as np
import cv2
try:
    import hdidx
except:
    import hdidx

from cbir.utils import dump,load


# generating sample data
ndim = 256     # dimension of features
ndb = 10000    # number of dababase items
nqry = 120     # number of queries

X_db = np.random.random((ndb, ndim)).astype(np.float64)
X_qry = np.random.random((nqry, ndim)).astype(np.float32)

## create Product Quantization Indexer
#idx = hdidx.indexer.PQIndexer()
## create Product Quantization Indexer
#idx = hdidx.indexer.PQIndexer()
## set storage for the indexer, this time we store the indexes into LMDB
#idx.set_storage('lmdb', {'path': './tmp/pq.idx'})
## build indexer
#idx.build({'vals': X_db, 'nsubq': 8})
#idx.add(X_db)
## save the index information for future use
#idx.save('./tmp/pq.info')

# create Product Quantization Indexer
idx = hdidx.indexer.PQIndexer()
# set storage for the indexer, this time we load the indexes from LMDB
idx.set_storage('lmdb', {'path': './tmp/pq.idx'})
# save the index information for future use
idx.load('./tmp/pq.info')
# searching in the database, and return top-100 items for each query
idx.search(X_qry, 100)