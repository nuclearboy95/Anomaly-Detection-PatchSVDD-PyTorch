import argparse
import matplotlib.pyplot as plt
from codes import mvtecad
from tqdm import tqdm
from codes.utils import resize, makedirpath


parser = argparse.ArgumentParser()
parser.add_argument('--obj', default='wood')
args = parser.parse_args()


def save_maps(obj, maps):
    from skimage.segmentation import mark_boundaries
    N = maps.shape[0]
    images = mvtecad.get_x(obj, mode='test')
    masks = mvtecad.get_mask(obj)

    for n in tqdm(range(N)):
        fig, axes = plt.subplots(ncols=2)
        fig.set_size_inches(6, 3)

        image = resize(images[n], (128, 128))
        mask = resize(masks[n], (128, 128))
        image = mark_boundaries(image, mask, color=(1, 0, 0), mode='thick')

        axes[0].imshow(image)
        axes[0].set_axis_off()

        axes[1].imshow(maps[n], vmax=maps[n].max(), cmap='Reds')
        axes[1].set_axis_off()

        plt.tight_layout()
        fpath = f'anomaly_maps/{obj}/n{n:03d}.png'
        makedirpath(fpath)
        plt.savefig(fpath)
        plt.close()


#########################


def main():
    from codes.inspection import eval_encoder_NN_multiK
    from codes.networks import EncoderHier

    obj = args.obj

    enc = EncoderHier(K=64, D=64).cuda()
    enc.load(obj)
    enc.eval()
    results = eval_encoder_NN_multiK(enc, obj)
    maps = results['maps_mult']

    save_maps(obj, maps)


if __name__ == '__main__':
    main()
