# -*- coding: utf-8 -*-

"""
Main executable for computing the memorization score

Author: G.J.J. van den Burg
License: See LICENSE file.
Copyright: 2021, The Alan Turing Institute

"""

import argparse
import gzip
import json
import os
import random

import torch

from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union
from typing import Optional

from dataset import SplitDataset
from dataset import KNOWN_DATASETS
from models import BaseModel
from models import MODELS
from models import load_model
from trainer import MarginalTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs"
    )
    parser.add_argument(
        "--dataset",
        choices=KNOWN_DATASETS,
        help="Dataset to use",
        default="MNIST",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Input batch size"
    )
    parser.add_argument(
        "--learning-rate",
        help="Learning rate for Adam",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODELS.keys()),
        help="Model to use",
        required=True,
    )
    parser.add_argument(
        "--latent-dim",
        help="Size of the latent dimension",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--repeats", help="Number of repetitions to run", default=5, type=int
    )
    parser.add_argument(
        "--result-dir",
        help="Directory to store output files",
        default="./results",
    )
    parser.add_argument(
        "--compute-px-every",
        help="Compute marginal every this many iterations",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--checkpoint-dir",
        help="Directory to store checkpoint files",
        default="./checkpoints",
    )
    parser.add_argument(
        "--checkpoint-every",
        help="Save a checkpoint every this many iterations",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-m",
        "--mode",
        help="Mode to run in",
        choices=["split-cv", "full"],
        default="split-cv",
    )
    parser.add_argument(
        "--start-at",
        help="Tuple (a, b) to start at run a fold b. Fold ignored when running in full mode",
        type=str,
        default="(0, 0)",
    )
    parser.add_argument(
        "--stop-after",
        help="Tuple (a, b) to stop after run a fold b. Fold ignored when running in full mode",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--num-feature",
        help="Feature map multiplier in DCVAE",
        type=int,
        default=32,
    )
    parser.add_argument("--seed", help="Random seed", type=int)
    return parser.parse_args()


def save_output(output: Dict[str, Any], filename: str, result_dir: str):
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, filename)
    with gzip.open(path, "wb") as fp:
        data = json.dumps(output, indent="\t", sort_keys=True)
        fp.write(str.encode(data))
    return filename


def get_filename(
    model: BaseModel,
    dataset: str,
    metadata: Dict[str, Union[int, float]],
    run: int,
    extension: str,
    mode: str = None,
    fold_idx: int = None,
    epoch: int = None,
):
    assert mode in ["cv", "full"]

    seed = metadata["seed"]
    base = f"{dataset}_{model.description}_seed{seed}"

    if mode == "cv":
        assert not fold_idx is None
        if epoch is None:
            filename = f"{base}_cv_repeat{run}_fold{fold_idx}.{extension}"
        else:
            filename = f"{base}_cv_repeat{run}_fold{fold_idx}_epoch{epoch}.{extension}"
    elif mode == "full":
        if epoch is None:
            filename = f"{base}_full_repeat{run}.{extension}"
        else:
            filename = f"{base}_full_repeat{run}_epoch{epoch}.{extension}"
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return filename


def store_result_split_cv(
    model: BaseModel,
    dataset: str,
    metadata: Dict[str, Union[int, float]],
    run: int,
    fold_idx: int,
    logpxs: dict,
    losses: dict,
    result_dir: str = "./results",
) -> str:
    filename = get_filename(
        model,
        dataset,
        metadata,
        run,
        extension="json.gz",
        mode="cv",
        fold_idx=fold_idx,
    )

    metadata = metadata.copy()
    metadata["model"] = model.description
    metadata["dataset"] = dataset
    metadata["run"] = run
    metadata["fold_idx"] = fold_idx

    output = dict(
        meta=metadata,
        results=dict(
            logpxs=logpxs,
            losses=losses,
        ),
    )
    return save_output(output, filename, result_dir)


def store_result_full(
    model: BaseModel,
    dataset: str,
    metadata: Dict[str, Union[int, float]],
    run: int,
    logpxs: dict,
    losses: dict,
    result_dir: str = "./results",
) -> str:
    filename = get_filename(
        model,
        dataset,
        metadata,
        run,
        extension="json.gz",
        mode="full",
    )

    metadata = metadata.copy()
    metadata["model"] = model.description
    metadata["dataset"] = dataset
    metadata["run"] = run

    output = dict(
        meta=metadata,
        results=dict(
            logpxs=logpxs,
            losses=losses,
        ),
    )
    return save_output(output, filename, result_dir)


def run_full(
    model_name: str,
    dataset: SplitDataset,
    repeats: int,
    compute_px_every: int,
    result_dir: str,
    checkpoint_dir: str,
    metadata: Dict[str, int],
    start_at: Tuple[int, int] = (-1, -1),
    stop_after: Optional[Tuple[int, int]] = None,
    checkpoint_every: Optional[int] = None,
) -> None:
    for run in range(repeats):
        if run < start_at[0]:
            print(f"Skipping {run=}")
            continue

        # Load a fresh instance of the model
        model = load_model(
            model_name,
            img_dim=metadata["img_dim"],
            latent_dim=metadata["latent_dim"],
            in_channels=dataset.channels,
            num_feature=metadata["num_feature"],
        )

        # Create a partial to generate checkpoint filenames
        chkp_func = lambda e: os.path.join(
            checkpoint_dir,
            get_filename(
                model,
                dataset.name,
                metadata,
                run,
                extension="params",
                mode="full",
                epoch=e,
            ),
        )

        trainer = MarginalTrainer(
            epochs=metadata["epochs"],
            lr=metadata["learning_rate"],
            checkpoint_func=chkp_func,
            params=metadata["trainer_params"],
        )
        in_loader = dataset.create_full()
        test_loader = dataset.create_full_test()
        logpxs, losses = trainer.fit(
            model,
            in_loader,
            test_loader,
            compute_px_every=compute_px_every,
            checkpoint_every=checkpoint_every,
            N_z=metadata["num_iw_samples"],
        )
        fname = store_result_full(
            model,
            dataset.name,
            metadata,
            run,
            logpxs,
            losses,
            result_dir=result_dir,
        )
        print(f"Stored results in: {fname}")

        if stop_after and run == stop_after[0]:
            return


def run_split_cv(
    model_name: str,
    dataset: SplitDataset,
    repeats: int,
    compute_px_every: int,
    result_dir: str,
    checkpoint_dir: str,
    metadata: Dict[str, int],
    start_at: Tuple[int, int] = (-1, -1),
    stop_after: Optional[Tuple[int, int]] = None,
    checkpoint_every: Optional[int] = None,
) -> None:

    for run in range(repeats):
        for fold_idx, (in_loader, out_loader) in enumerate(
            dataset.create_splits_cv(run, folds=10)
        ):
            if run < start_at[0]:
                print(f"Skipping {run=} and {fold_idx=}")
                continue

            if run == start_at[0] and fold_idx < start_at[1]:
                print(f"Skipping {run=} and {fold_idx=}")
                continue

            # Load a fresh instance of the model
            model = load_model(
                model_name,
                img_dim=metadata["img_dim"],
                latent_dim=metadata["latent_dim"],
                in_channels=dataset.channels,
                num_feature=metadata["num_feature"],
            )

            # Create a partial to generate checkpoint filenames
            chkp_func = lambda e: os.path.join(
                checkpoint_dir,
                get_filename(
                    model,
                    dataset.name,
                    metadata,
                    run,
                    extension="params",
                    mode="cv",
                    fold_idx=fold_idx,
                    epoch=e,
                ),
            )

            # Instantiate trainer
            trainer = MarginalTrainer(
                epochs=metadata["epochs"],
                lr=metadata["learning_rate"],
                checkpoint_func=chkp_func,
                params=metadata["trainer_params"],
            )

            # Train the model
            logpxs, losses = trainer.fit(
                model,
                in_loader,
                out_loader,
                compute_px_every=compute_px_every,
                checkpoint_every=checkpoint_every,
                N_z=metadata["num_iw_samples"],
            )

            # Store the results
            fname = store_result_split_cv(
                model,
                dataset.name,
                metadata,
                run,
                fold_idx,
                logpxs,
                losses,
                result_dir=result_dir,
            )
            print(f"Stored results in: {fname}")

            if (
                stop_after
                and run == stop_after[0]
                and fold_idx == stop_after[1]
            ):
                return


def main():
    args = parse_args()

    seed = args.seed or random.randint(1000, 10000)
    torch.manual_seed(seed)
    print(f"Running with seed: {seed}")

    img_dim = 32

    dataset_params = {}
    if args.model in ["DiagonalGaussianDCVAE", "ConstantGaussianDCVAE"]:
        dataset_params["dequantize"] = True
        dataset_params["logits"] = True
    elif args.dataset in ["BinarizedMNIST"]:
        dataset_params["binarize"] = True

    dataset = SplitDataset(
        args.dataset,
        seed,
        img_dim=img_dim,
        batch_size=args.batch_size,
        params=dataset_params,
    )

    trainer_params = {}
    if args.dataset in ["CIFAR10", "CelebA"]:
        trainer_params["hflip"] = True

    if args.dataset in ["CIFAR10", "CelebA"]:
        num_feature = args.num_feature
    elif args.model.endswith("DCVAE"):
        num_feature = args.num_feature
    else:
        num_feature = None

    metadata = dict(
        seed=seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        img_dim=img_dim,
        latent_dim=args.latent_dim,
        mode=args.mode,
        num_iw_samples=256 if "MNIST" in args.dataset else 128,
        trainer_params=trainer_params,
        num_feature=num_feature,
    )

    parse_tup = lambda s: list(map(int, s.strip("()").split(",")))
    start_at = parse_tup(args.start_at)
    stop_after = parse_tup(args.stop_after) if args.stop_after else None

    if args.mode == "split-cv":
        runner = run_split_cv
    elif args.mode == "full":
        runner = run_full
    else:
        raise ValueError(f"Unknown operating mode: {args.mode}")

    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    runner(
        args.model,
        dataset,
        args.repeats,
        args.compute_px_every,
        args.result_dir,
        args.checkpoint_dir,
        metadata,
        start_at=start_at,
        stop_after=stop_after,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
