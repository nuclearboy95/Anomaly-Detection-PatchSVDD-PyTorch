import numpy as np
from torch.utils.data import Dataset
from .utils import *


__all__ = ['SVDD_Dataset', 'PositionDataset']


def generate_coords(H, W, K):
    h = np.random.randint(0, H - K + 1)
    w = np.random.randint(0, W - K + 1)
    return h, w


def generate_coords_position(H, W, K):
    with task('P1'):
        p1 = generate_coords(H, W, K)
        h1, w1 = p1

    pos = np.random.randint(8)

    with task('P2'):
        J = K // 4

        K3_4 = 3 * K // 4
        h_dir, w_dir = pos_to_diff[pos]
        h_del, w_del = np.random.randint(J, size=2)

        h_diff = h_dir * (h_del + K3_4)
        w_diff = w_dir * (w_del + K3_4)

        h2 = h1 + h_diff
        w2 = w1 + w_diff

        h2 = np.clip(h2, 0, H - K)
        w2 = np.clip(w2, 0, W - K)

        p2 = (h2, w2)

    return p1, p2, pos


def generate_coords_svdd(H, W, K):
    with task('P1'):
        p1 = generate_coords(H, W, K)
        h1, w1 = p1

    with task('P2'):
        J = K // 32

        h_jit, w_jit = 0, 0

        while h_jit == 0 and w_jit == 0:
            h_jit = np.random.randint(-J, J + 1)
            w_jit = np.random.randint(-J, J + 1)

        h2 = h1 + h_jit
        w2 = w1 + w_jit

        h2 = np.clip(h2, 0, H - K)
        w2 = np.clip(w2, 0, W - K)

        p2 = (h2, w2)

    return p1, p2


pos_to_diff = {
    0: (-1, -1),
    1: (-1, 0),
    2: (-1, 1),
    3: (0, -1),
    4: (0, 1),
    5: (1, -1),
    6: (1, 0),
    7: (1, 1)
}


class SVDD_Dataset(Dataset):
    def __init__(self, memmap, K=64, repeat=1):
        super().__init__()
        self.arr = np.asarray(memmap)
        self.K = K
        self.repeat = repeat

    def __len__(self):
        N = self.arr.shape[0]
        return N * self.repeat

    def __getitem__(self, idx):
        N = self.arr.shape[0]
        K = self.K
        n = idx % N

        p1, p2 = generate_coords_svdd(256, 256, K)

        image = self.arr[n]

        patch1 = crop_image_CHW(image, p1, K)
        patch2 = crop_image_CHW(image, p2, K)

        return patch1, patch2

    @staticmethod
    def infer(enc, batch):
        x1s, x2s, = batch
        h1s = enc(x1s)
        h2s = enc(x2s)
        diff = h1s - h2s
        l2 = diff.norm(dim=1)
        loss = l2.mean()

        return loss


class PositionDataset(Dataset):
    def __init__(self, x, K=64, repeat=1):
        super(PositionDataset, self).__init__()
        self.x = np.asarray(x)
        self.K = K
        self.repeat = repeat

    def __len__(self):
        N = self.x.shape[0]
        return N * self.repeat

    def __getitem__(self, idx):
        N = self.x.shape[0]
        K = self.K
        n = idx % N

        image = self.x[n]
        p1, p2, pos = generate_coords_position(256, 256, K)

        patch1 = crop_image_CHW(image, p1, K).copy()
        patch2 = crop_image_CHW(image, p2, K).copy()

        # perturb RGB
        rgbshift1 = np.random.normal(scale=0.02, size=(3, 1, 1))
        rgbshift2 = np.random.normal(scale=0.02, size=(3, 1, 1))

        patch1 += rgbshift1
        patch2 += rgbshift2

        # additive noise
        noise1 = np.random.normal(scale=0.02, size=(3, K, K))
        noise2 = np.random.normal(scale=0.02, size=(3, K, K))

        patch1 += noise1
        patch2 += noise2

        return patch1, patch2, pos
