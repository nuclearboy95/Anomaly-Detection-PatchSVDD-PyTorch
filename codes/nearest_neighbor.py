import numpy as np
import shutil
import os


__all__ = ['search_NN']


def search_NN(test_emb, train_emb_flat, NN=1, method='kdt'):
    if method == 'ngt':
        return search_NN_ngt(test_emb, train_emb_flat, NN=NN)

    from sklearn.neighbors import KDTree
    kdt = KDTree(train_emb_flat)

    Ntest, I, J, D = test_emb.shape
    closest_inds = np.empty((Ntest, I, J, NN), dtype=np.int32)
    l2_maps = np.empty((Ntest, I, J, NN), dtype=np.float32)

    for n in range(Ntest):
        for i in range(I):
            dists, inds = kdt.query(test_emb[n, i, :, :], return_distance=True, k=NN)
            closest_inds[n, i, :, :] = inds[:, :]
            l2_maps[n, i, :, :] = dists[:, :]

    return l2_maps, closest_inds


def search_NN_ngt(test_emb, train_emb_flat, NN=1):
    import ngtpy

    Ntest, I, J, D = test_emb.shape
    closest_inds = np.empty((Ntest, I, J, NN), dtype=np.int32)
    l2_maps = np.empty((Ntest, I, J, NN), dtype=np.float32)

    # os.makedirs('tmp', exist_ok=True)
    dpath = f'/tmp/{os.getpid()}'
    ngtpy.create(dpath, D)
    index = ngtpy.Index(dpath)
    index.batch_insert(train_emb_flat)

    for n in range(Ntest):
        for i in range(I):
            for j in range(J):
                query = test_emb[n, i, j, :]
                results = index.search(query, NN)
                inds = [result[0] for result in results]

                closest_inds[n, i, j, :] = inds
                vecs = np.asarray([index.get_object(inds[nn]) for nn in range(NN)])
                dists = np.linalg.norm(query - vecs, axis=-1)
                l2_maps[n, i, j, :] = dists
    shutil.rmtree(dpath)

    return l2_maps, closest_inds
