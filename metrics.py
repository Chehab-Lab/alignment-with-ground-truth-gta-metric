import numpy as np
import faiss
from enum import Enum
import pdb, os
from scipy.stats import entropy
import gc
from scipy.cluster.hierarchy import linkage, cut_tree

class Metric(Enum):
    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"

def gta_kmeans(embeddings, labels, ks):
    
    labels = np.asarray(labels)
    embeddings = np.asarray(embeddings)
    
    X = np.asarray(embeddings)
    Y = np.asarray(labels)
    N = len(Y)

    del embeddings
    gc.collect()

    label_counts = np.bincount(Y.astype(int))
    H_Y = entropy(label_counts)

    if H_Y == 0:
        return {k: 1.0 for k in ks}

    alignments = {}
    for k in ks:
        kmeans = faiss.Kmeans(
            d=X.shape[1],
            k=k,
            niter=20,
            seed=42,
            verbose=False,
        )
        kmeans.train(X)
        cluster_ids = kmeans.index.search(X, 1)[1].flatten()

        H_Y_given_C = 0.0
        for c in range(k):
            
            idx = np.where(cluster_ids == c)[0]
            if len(idx) == 0: continue
            
            labels_c = Y[idx]
            counts_c = np.bincount(labels_c.astype(int))
            
            H_c = entropy(counts_c)
            H_Y_given_C += (len(idx) / N) * H_c

        alignment = 1.0 - H_Y_given_C / H_Y
        alignments[k] = max(0.0, min(1.0, alignment))

    del X
    del Y
    gc.collect()

    return alignments


def gta_hierarchical(embeddings, labels, ks):
    
    labels = np.asarray(labels)
    embeddings = np.asarray(embeddings)
    
    X = np.asarray(embeddings)
    Y = np.asarray(labels)
    N = len(Y)

    del embeddings
    gc.collect()

    label_counts = np.bincount(Y.astype(int))
    H_Y = entropy(label_counts)

    if H_Y == 0:
        return {k: 1.0 for k in ks}

    Z = linkage(X, method='ward')

    alignments = {}
    for k in ks:
        cluster_ids = cut_tree(Z, n_clusters=k).flatten()

        H_Y_given_C = 0.0
        for c in np.unique(cluster_ids):
            
            idx = np.where(cluster_ids == c)[0]
            if len(idx) == 0: continue
            
            labels_c = Y[idx]
            counts_c = np.bincount(labels_c.astype(int))
            
            H_c = entropy(counts_c)
            H_Y_given_C += (len(idx) / N) * H_c

        alignment = 1.0 - H_Y_given_C / H_Y
        alignments[k] = max(0.0, min(1.0, alignment))

    del X
    del Y
    del Z
    gc.collect()

    return alignments
