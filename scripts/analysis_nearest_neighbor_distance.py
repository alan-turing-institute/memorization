# -*- coding: utf-8 -*-

"""
Compute nearest neighbor distances.

Author: G.J.J. van den Burg
License: See LICENSE file.
Copyright: 2021, The Alan Turing Institute

"""

import argparse
import torch
import numpy as np

from tqdm import trange
from torch.distributions import MultivariateNormal
from torchvision.transforms.functional import resize

from constants import LOGIT_LAMBDA

from dataset import KNOWN_DATASETS
from dataset import SplitDataset

from models import MODELS
from models import load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Checkpoint file", required=True)
    parser.add_argument(
        "--model", help="Model name", choices=MODELS, required=True
    )
    parser.add_argument(
        "--dataset", help="Dataset name", choices=KNOWN_DATASETS, required=True
    )
    parser.add_argument(
        "--latent-dim", help="latent dim", type=int, required=True
    )
    parser.add_argument("--seed", help="Random seed for sampling", type=int)
    parser.add_argument(
        "-o", "--output", help="Output file for distance results (.npz)"
    )
    return parser.parse_args()


def init_model(
    model_name, checkpoint_file, latent_dim, in_channels, img_dim=32
):
    model = load_model(
        model_name,
        img_dim,
        latent_dim=latent_dim,
        in_channels=in_channels,
    )
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def resize_cdist(A, B, frac=0.5):
    size = int(A.shape[2] * frac)
    AA = resize(A, (size, size)).reshape(A.shape[0], -1)
    BB = resize(B, (size, size)).reshape(B.shape[0], -1)
    return torch.cdist(AA, BB)


def unlogit(x):
    return (torch.sigmoid(x) - LOGIT_LAMBDA) / (1.0 - 2 * LOGIT_LAMBDA)


def main():
    args = parse_args()
    seed = args.seed if args.seed else np.random.randint(10000)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.autograd.set_grad_enabled(False)

    img_dim = 32
    L = args.latent_dim
    logit_datasets = ["CIFAR10", "CelebA"]
    logit_model = args.dataset in logit_datasets

    sd = SplitDataset(
        args.dataset,
        args.seed,
        img_dim=img_dim,
        batch_size=64,
        params={"logits": False},
    )
    model = init_model(
        args.model, args.checkpoint, L, sd.channels, img_dim=img_dim
    )

    n_train = len(sd._data)
    valdata = sd._valdata if hasattr(sd, "_valdata") else sd._testdata
    n_val = len(valdata)

    # Load dataset into memory (for convenience)
    X = torch.zeros(n_train, sd.channels, img_dim, img_dim)
    for i in trange(n_train, desc="Loading dataset"):
        X[i] = sd.get_item_by_index(i)[0]

    # Load validation set into memory
    V = torch.zeros((n_val, sd.channels, img_dim, img_dim))
    for i in trange(n_val, desc="loading val"):
        V[i] = valdata[i][0]

    # Generate samples
    mvn0 = MultivariateNormal(torch.zeros(L), torch.eye(L))
    Z = mvn0.sample((n_val,))
    S = model.push(Z)
    S = unlogit(S) if logit_model else S

    # Find nearest neighbors
    nn_val = torch.zeros((n_train,), dtype=int)
    nn_sample = torch.zeros((n_train,), dtype=int)
    dist_val = torch.zeros((n_train,))
    dist_sample = torch.zeros((n_train,))

    bs = 256
    for s in trange(0, n_train, bs, desc="Finding NNs (val)"):
        dd = resize_cdist(X[s : s + bs], V)
        idxs = torch.argmin(dd, axis=1)
        nn_val[s : s + bs] = idxs
        dist_val[s : s + bs] = dd[0, idxs]

    for s in trange(0, n_train, bs, desc="Finding NNs (sample)"):
        dd = resize_cdist(X[s : s + bs], S)
        idxs = torch.argmin(dd, axis=1)
        nn_sample[s : s + bs] = idxs
        dist_sample[s : s + bs] = dd[0, idxs]

    nn_val = nn_val.numpy()
    nn_sample = nn_sample.numpy()
    dist_val = dist_val.numpy()
    dist_sample = dist_sample.numpy()
    ratio = dist_val / dist_sample

    np.savez(
        args.output,
        seed=args.seed,
        nn_val=nn_val,
        nn_sample=nn_sample,
        dist_val=dist_val,
        dist_sample=dist_sample,
        ratio=ratio,
    )


if __name__ == "__main__":
    main()
