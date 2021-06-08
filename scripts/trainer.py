# -*- coding: utf-8 -*-

"""
Functionality to train a model on a data set.

We use a Trainer class to train a model that follows the BaseModel/BaseVAE abc 
defined in the model.py file.

Author: G.J.J. van den Burg
License: See LICENSE file.
Copyright: 2021, The Alan Turing Institute

"""

import abc
import time
import torch

from collections import defaultdict

from typing import Any
from typing import Dict
from typing import Optional

from tqdm import tqdm

from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader
from torchvision.transforms.functional import hflip

from models import BaseModel
from models import BaseVAE


class BaseTrainer(metaclass=abc.ABCMeta):
    def __init__(
        self,
        device=None,
        epochs=20,
        lr=1e-3,
        checkpoint_func=None,
        params: Optional[Dict[str, Any]] = None,
    ):
        self._cuda = torch.cuda.is_available()
        self._device = device or torch.device("cuda" if self._cuda else "cpu")
        self._epochs = epochs
        self._lr = lr
        self._print_every = 10
        self._checkpoint_func = checkpoint_func
        self._params = params or {}

    @abc.abstractmethod
    def fit(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader],
        **kwargs,
    ):
        """Train a model on a given data set"""

    def train(self, model: BaseModel, loader: DataLoader, epoch: int):
        total_loss = 0.0

        model.train()
        for b, (batch_x, _, batch_idxs) in enumerate(loader):
            # This is here and not in the data loader to ensure it is not
            # applied to the test set or the marginal likelihood computation
            if "hflip" in self._params and self._params["hflip"]:
                batch_x = hflip(batch_x) if torch.rand(1) < 0.5 else batch_x
            batch_x = batch_x.to(self._device)

            self._optimizer.zero_grad()
            loss = model.step(batch_x)
            loss.backward()
            self._optimizer.step()

            total_loss += loss.item()

            if b % 10 == 0:
                print(
                    f"Train epoch: {epoch} "
                    f"[{b * len(batch_x)}/{len(loader.dataset)} "
                    f"({(100.  * b / len(loader)):.2f})%]"
                    f"\tLoss: {loss.item()/len(batch_x):.4f}"
                )

        avg_loss = total_loss / len(loader.dataset)
        print(f"=====> Epoch: {epoch}. Average train loss: {avg_loss:.4f}")
        return avg_loss

    def test(self, model: BaseModel, loader: DataLoader, epoch: int):
        total_loss = 0.0

        model.eval()
        with torch.no_grad():
            for b, (batch_x, _, batch_idxs) in enumerate(loader):
                batch_x = batch_x.to(self._device)
                loss = model.step(batch_x)

                total_loss += loss.item()

        avg_loss = total_loss / len(loader.dataset)
        print(f"=====> Epoch: {epoch}. Average test  loss: {avg_loss:.4f}")
        return avg_loss


class MarginalTrainer(BaseTrainer):
    def fit(
        self,
        model: BaseVAE,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader],
        compute_px_every=10,
        checkpoint_every=None,
        N_z: int = 256,
    ):
        model.to(self._device)
        self._optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)

        logpxs = defaultdict(dict)
        losses = defaultdict(list)

        for epoch in range(1, self._epochs + 1):
            t0 = time.time()
            loss_train = self.train(model, train_loader, epoch)
            if test_loader:
                loss_test = self.test(model, test_loader, epoch)
            t1 = time.time()
            print(f"=====> Epoch: {epoch}.           Duration: {(t1-t0):.4f}")

            if compute_px_every and epoch % compute_px_every == 0:
                logpxs[epoch]["train"] = self.marginal(
                    model, train_loader, N=N_z
                )
                if test_loader:
                    logpxs[epoch]["test"] = self.marginal(
                        model, test_loader, N=N_z
                    )

            losses["train"].append(loss_train)
            if test_loader:
                losses["test"].append(loss_test)

            if checkpoint_every and epoch % checkpoint_every == 0:
                param_file = self._checkpoint_func(epoch)
                torch.save(model.state_dict(), param_file)
        return logpxs, losses

    def marginal(self, model: BaseModel, loader: DataLoader, N: int = 256):
        logpxs = None
        idxs = None

        model.eval()
        with torch.no_grad():
            for b, (X, Y, I) in tqdm(enumerate(loader), total=len(loader)):
                X = X.to(model._device)
                logpx_batch = self.marginal_batch(model, X, N=N)

                if logpxs is None:
                    logpxs = logpx_batch
                    idxs = I
                else:
                    logpxs = torch.cat((logpxs, logpx_batch))
                    idxs = torch.cat((idxs, I))

        logpxs = list(map(float, list(logpxs.cpu().numpy())))
        idxs = list(map(int, list(idxs.cpu().numpy())))
        return dict(zip(idxs, logpxs))

    def marginal_batch(
        self, model: BaseVAE, batch: torch.Tensor, N: int = 256
    ) -> torch.Tensor:

        L = model.latent_dim
        B = batch.shape[0]
        dev = batch.device

        X = batch

        mu0 = torch.zeros(L, device=dev)
        cov0 = torch.eye(L, device=dev)
        mvn0 = MultivariateNormal(mu0, cov0)

        mu_x, logvar_x = model.encode(X)
        var_x = logvar_x.exp()
        mvn1 = MultivariateNormal(mu_x, torch.diag_embed(var_x))

        zs = mvn1.sample((N,))

        logz0 = mvn0.log_prob(zs)
        logz1 = mvn1.log_prob(zs)

        Z = zs.view(-1, L)

        Y = model.decode(Z)
        X = X.repeat(N, 1, 1, 1)

        del Z

        logpxz = model.log_pxz(Y, X)
        logpxz = logpxz.view(N, B)

        v = logpxz + logz0 - logz1

        log_px = -torch.log(torch.tensor(float(N))) + torch.logsumexp(v, 0)
        return log_px
