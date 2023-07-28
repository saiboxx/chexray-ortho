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
MODEL_ID = 'densenet121-res224-chex'
EMBEDDING_FILE = 'chex_densenet_chex.npy'
DEVICE = 'cuda'


@torch.no_grad()
def main() -> None:
    device = torch.device(DEVICE)
    dfs = get_chexpert_meta_data(DATA_DIR)

    transform = Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
    model = xrv.models.get_model(MODEL_ID).to(device)
    model.eval()

    num_samples = sum([len(d) for d in dfs])
    emb_dim = get_features(torch.rand(1, 1, 224, 224).to(device), model).shape[0]
    print(f'Obtaining embeddings of dimension {emb_dim} for {num_samples} samples')

    emb = np.zeros((num_samples, emb_dim), dtype=float)

    for j, df in enumerate(dfs, start=1):
        print(f'Dataframe {j} of {len(dfs)}:')
        for i, row in tqdm(df.iterrows(), total=len(df)):
            f_path = row['Path']
            img = get_image(f_path, transform).to(device)
            emb[row['index']] = get_features(img, model)

    np.save(emb)


def get_image(f_path: str, transform: Any = None) -> Tensor:
    img = skimage.io.imread(f_path)
    img = xrv.datasets.normalize(img, 255)
    img = img.mean(2)[None, ...]

    if transform is not None:
        img = transform(img)
    return torch.from_numpy(img)


def get_features(x: Tensor, model: nn.Module) -> np.ndarray:
    feats = model.features(x)
    feats = F.relu(feats, inplace=True)
    feats = F.adaptive_avg_pool2d(feats, (1, 1))
    return feats.flatten().cpu().numpy()


if __name__ == '__main__':
    main()
