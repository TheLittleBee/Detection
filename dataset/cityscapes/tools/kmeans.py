import numpy as np


def IoU(x, y):
    minw = min(x[0], y[0])
    minh = min(x[1], y[1])
    inter = minw * minh
    union = x[0] * x[1] + y[0] * y[1] - inter
    return float(inter) / union


class KMeans():
    def __init__(self, n_clusters, max_iter=300):
        self.k = n_clusters
        self.max_iter = max_iter

    def fix(self, X, init=True):
        if init:
            self._k_init(X)
        else:
            self._init(X)
        self.labels = np.zeros(X.shape[0])
        for i in range(self.max_iter):
            loss, change = self._assign(X)
            if not change: break
            self._update(X)
        self.loss = loss
        return self

    def _distance(self, x, centers):
        res = 1
        ind = -1
        for i, center in enumerate(centers):
            dis = 1 - IoU(x, center)
            if dis < res:
                res = dis
                ind = i
        return res, ind

    def _k_init(self, X):
        n_samples, n_features = X.shape
        centers = np.empty((self.k, n_features), dtype=X.dtype)
        center_id = np.random.randint(n_samples)
        centers[0] = X[center_id]
        for i in range(1, self.k):
            distances = [self._distance(x, centers[:i])[0] for x in X]
            total = np.sum(distances)
            rand = np.random.rand() * total
            center_id = np.searchsorted(np.cumsum(distances), rand)
            centers[i] = X[center_id]
        self.centers = centers

    def _init(self, X):
        seeds = np.random.permutation(X.shape[0])[:self.k]
        self.centers = X[seeds].copy()

    def _assign(self, X):
        inertia = 0.
        change = 0
        for i, x in enumerate(X):
            dis, ind = self._distance(x, self.centers)
            if ind != self.labels[i]:
                self.labels[i] = ind
                change += 1
            inertia += dis
        return inertia, change

    def _update(self, X):
        for i in range(self.k):
            x = X[self.labels == i]
            self.centers[i] = np.mean(x, axis=0)


if __name__ == '__main__':
    from pycocotools.coco import COCO
    import matplotlib.pyplot as plt

    coco = COCO(
        '/home/littlebee/code/pytorch/Detection/data/Cityscapes/annotations/instancesonly_filtered_gtFine_train.json')
    anno_ids = coco.getAnnIds()
    annos = coco.loadAnns(anno_ids)
    X = np.zeros((len(annos), 2))
    for i, anno in enumerate(annos):
        X[i, 0] = anno['bbox'][2]
        X[i, 1] = anno['bbox'][3]
    X[:, 0] *= 640 / 2048
    X[:, 1] *= 320 / 1024
    X = X[X[:, 0] > 16]
    X = X[X[:, 1] > 16]
    X = X[X[:, 0] < 300]
    X = X[X[:, 1] < 300]
    print(X.shape)

    k = 9
    kmeans = KMeans(k)
    kmeans.fix(X)
    print(kmeans.centers)
    l = kmeans.labels
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'gold', 'purple']
    for i in range(k):
        plt.scatter(X[l == i, 0], X[l == i, 1], s=10, c=color[i], marker='o')
    plt.scatter(kmeans.centers[:, 0], kmeans.centers[:, 1], s=100, c='gray', marker='+')
    plt.show()
