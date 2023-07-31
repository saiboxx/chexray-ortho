import os
from multiprocessing import Pool
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.transforms import Compose
import torchxrayvision as xrv
import skimage, skimage.io
from tqdm import tqdm

from src.utils import get_chexpert_meta_data

DATA_DIR = 'data'
CHEST_DIR = '/data/core-rad/data/chestx-ray'
MODEL_ID = 'densenet121-res224-chex'
EMBEDDING_FILE = 'chex_densenet_chex.npy'
DEVICE = 'cuda'
BATCH_SIZE = 1024
POOL_WORKERS = 64


@torch.no_grad()
def main() -> None:
    device = torch.device(DEVICE)
    dfs = get_chexpert_meta_data(DATA_DIR)

    transform = Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
    model = xrv.models.get_model(MODEL_ID).to(device)
    model.eval()

    num_samples = sum([len(d) for d in dfs])
    emb_dim = get_features(torch.rand(1, 1, 224, 224).to(device), model).shape[1]
    print(f'Obtaining embeddings of dimension {emb_dim} for {num_samples} samples')

    emb = np.zeros((num_samples, emb_dim), dtype=float)

    with Pool(processes=POOL_WORKERS) as pool:
        for j, df in enumerate(dfs, start=1):
            print(f'Dataframe {j} of {len(dfs)}:')

            df_chunked = [df[i : i + BATCH_SIZE] for i in range(0, len(df), BATCH_SIZE)]
            img_batch = []
            idxs = []
            load_func = partial(get_image, transform=transform)

            for chunk in tqdm(df_chunked):
                idxs.extend(chunk['idx'].tolist())
                arg = [os.path.join(CHEST_DIR, p) for p in chunk['Path'].tolist()]
                img_batch.extend(list(pool.imap(load_func, arg, chunksize=32)))

                img_tens = torch.from_numpy(np.stack(img_batch)).to(device)
                img_feats = get_features(img_tens, model)

                for i in range(len(idxs)):
                    emb[idxs[i], :] = img_feats[i]

                img_batch.clear()
                idxs.clear()

    emp_path = os.path.join(DATA_DIR, EMBEDDING_FILE)
    np.save(emp_path, emb)


def get_image(f_path: str, transform: Any = None) -> np.ndarray:
    img = skimage.io.imread(f_path, as_gray=True)
    img = xrv.datasets.normalize(img, 255)
    img = img[None, ...]

    if transform is not None:
        img = transform(img)
    return img


def get_features(x: Tensor, model: nn.Module) -> np.ndarray:
    if len(x.shape) == 3:
        x = x.unsqueeze(0)
    feats = model.features(x)
    feats = F.relu(feats, inplace=True)
    feats = F.adaptive_avg_pool2d(feats, (1, 1))
    return feats.view(x.shape[0], -1).cpu().numpy()


if __name__ == '__main__':
    main()
