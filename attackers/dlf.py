"""
V. Shejwalkar, A. Houmansadr, P. Kairouz, and D. Ramage, “Back to the Drawing Board: A Critical Evaluation of
Poisoning Attacks on Production Federated Learning.” arXiv, Dec. 13, 2021. doi: 10.48550/arXiv.2108.10241.
"""
from __future__ import annotations

import copy
import itertools
import random
from typing import Sized, Iterable

import torch
import torch.distributions as distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import flcore
import flcore.utils.model as model_utils
from attackers.core import Attacker, AttackInfo

__all__ = ["DlfAttacker"]


class DlfAttacker(Attacker):
    _partners: dict[str, DlfAttacker] = {}
    _dlf_model: nn.Module = None

    def __init__(self, *, client: flcore.Client, dlf_size: int, noise_std: float, **kwargs):
        super().__init__(client=client, **kwargs)
        self._dlf_size = dlf_size
        self._noise_std = noise_std
        self._partners[self._client.id] = self

        if self._dlf_model is None:
            self.__class__._dlf_model = copy.deepcopy(self._client.model)

    def attack_algorithm(self, info: AttackInfo):
        self.receive_model(info.global_model)

        for local_epoch in range(self.max_epoch):
            self.train(self._dlf_dataloader)

        self.attack_log["dlf_size"] = self._dlf_size
        self.attack_log["cosine"] = float(F.cosine_similarity(
            model_utils.model_to_vector(self.model), info.global_vector, dim=0
        ))
        self.attack_log["l2_dist"] = float(model_utils.model_to_vector(self.model).dist(info.global_vector, p=2))

    @property
    @torch.no_grad()
    def _dlf_dataloader(self) -> data.DataLoader:
        # generate a model to evaluate
        model_utils.move_parameters(self.model, self._dlf_model)
        # gather all dataset we have
        datasets: list[data.Dataset | Sized] = [partner.train_dataset for partner in self._partners.values()]
        # calculate how many data we have
        dataset_size = sum(map(len, datasets))
        # add noise if our data is not enough
        noise = True if self._dlf_size > dataset_size else False
        # shuffle to make sure gradients are different when more than two attackers launch an attack at the same time
        random.shuffle(datasets)

        dlf_dataset = DlfDataset(data.ConcatDataset(datasets), self._dlf_model,
                                 size=self._dlf_size,
                                 noise=noise,
                                 std=self._noise_std)
        batch_size = self.train_dataloader.batch_size
        dlf_dataloader = data.DataLoader(dlf_dataset, batch_size=batch_size)

        self.attack_log["noise_std"] = self._noise_std if noise else 0

        return dlf_dataloader


class DlfDataset(data.IterableDataset):
    def __getitem__(self, index):
        return super().__getitem__(index)

    def __init__(self, dataset: data.Dataset | Iterable, model: nn.Module, size: int, noise: bool, std: float):
        super().__init__()
        self.dataset = dataset
        self.size = size
        self.noise = noise
        self.model = model

        if noise:
            self.m = distributions.Normal(0, std)

        self.model.cuda()

    @torch.no_grad()
    def __iter__(self):
        self.model.eval()

        for i, (x, _) in enumerate(itertools.cycle(self.dataset)):
            if i >= self.size:
                return

            if self.noise:
                x.add_(self.m.sample(x.size()))

            batch_x = x.view(1, *x.size()).cuda()
            y_bad = self.model(batch_x).argmin()

            yield x, y_bad

    def __len__(self):
        return self.size
