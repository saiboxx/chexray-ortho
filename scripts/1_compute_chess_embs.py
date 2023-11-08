import os
from multiprocessing import Pool
from functools import partial
from typing import Any

import numpy as np
import torch
from torch import nn
from torchvision.transforms import Compose
import torchxrayvision as xrv
import skimage, skimage.io
from tqdm import tqdm
from torch.nn import Conv2d, Identity
from torchvision.models import resnet50

from src.utils import get_chexpert_meta_data, get_mimic_meta_data

DATASET = 'chex'
DATA_DIR = 'data'
CHEST_DIR = '<path to containing MIMIC and CHEXPERT>'
CHESS_PATH = 'data/chess.pt'
EMBEDDING_FILE = 'chex_chess.npy'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 256
POOL_WORKERS = 32


@torch.no_grad()
def main() -> None:
    device = torch.device(DEVICE)

    if DATASET == 'mimic':
        dfs = get_mimic_meta_data(os.path.join(DATA_DIR, 'mimic_meta.csv'))
        for df in dfs:
            df['Path'] = (
                'mimic-cxr-jpg-2.0.0'
                + '/files'
                + '/p'
                + df['subject_id'].astype(str).str[:2]
                + '/p'
                + df['subject_id'].astype(str)
                + '/s'
                + df['study_id'].astype(str)
                + '/'
                + df['dicom_id']
                + '.jpg'
            )
    else:
        dfs = get_chexpert_meta_data(DATA_DIR)

    transform = Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(512)])
    model = get_model().to(device)
    model.eval()

    num_samples = sum([len(d) for d in dfs])
    emb_dim = model(torch.rand(1, 1, 512, 512).to(device)).shape[1]
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
                img_feats = model(img_tens).cpu().numpy()

                for i in range(len(idxs)):
                    emb[idxs[i], :] = img_feats[i]

                img_batch.clear()
                idxs.clear()

    emp_path = os.path.join(DATA_DIR, EMBEDDING_FILE)
    np.save(emp_path, emb)


def get_model() -> nn.Module:
    model = resnet50()
    model.conv1 = Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    model.fc = Identity()
    state_dict = torch.load(CHESS_PATH, map_location='cpu')
    model.load_state_dict(state_dict)

    for name, param in model.named_parameters():
        param.requires_grad = False

    return model


def get_image(f_path: str, transform: Any = None) -> np.ndarray:
    img = skimage.io.imread(f_path, as_gray=True)
    img = img[None, ...]
    if transform is not None:
        img = transform(img)

    img = np.clip(img, 0, np.percentile(img, 99))
    img -= img.min()
    img /= img.max() - img.min()

    return (img - 0.658) / 0.221


if __name__ == '__main__':
    main()
