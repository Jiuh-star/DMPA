"""
V. Shejwalkar and A. Houmansadr,
“Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning,”
in Proceedings 2021 Network and Distributed System Security Symposium,
Virtual, 2021. doi: 10.14722/ndss.2021.24498.
"""
from __future__ import annotations

import copy
from typing import Sequence, Literal

import torch
import torch.nn as nn

import flcore
import flcore.utils.model as model_utils
import flcore.utils.robust as robust
from .lmpa import bisect_right
from attackers.core import Attacker, AttackInfo

__all__ = ["AgrTailoredAttacker", "MinMaxAttacker", "MinSumAttacker"]


class AgrTailoredAttacker(Attacker):
    _buffer: nn.Module = None

    def __init__(self, *, client: flcore.Client, perturb_vector: Literal["unit", "std", "sign"] | str,
                 max_scale: float | int, **kwargs):
        super().__init__(client=client, **kwargs)
        assert perturb_vector in ["unit", "std", "sign"]

        self._perturb_vector = perturb_vector
        self._min_lambda = 1E-3
        self._max_lambda = max_scale  # limiting toxicity since we are under non-iid.
        self._prev_loss = - 1

    def attack_algorithm(self, info: AttackInfo):
        if self._buffer is None:
            self.__class__._buffer = copy.deepcopy(self.model).to(self.device)
            setattr(self.__class__._buffer, "attack_flag", True)

        self.receive_model(info.global_model)

        # get deviation
        deviation = self._perturb(info.reference_update, info.benign_updates)

        # We don't use the original algorithm so that we can control the max toxicity.
        lambda_ = self._max_lambda
        self._prev_loss = -1

        if info.robust_fn:
            index = bisect_right(self._lambda_range, bypass=lambda x: self._bypass(x, deviation, info))
            lambda_ = self._lambda_range[index]

        model_utils.vector_to_model(info.global_vector + info.reference_update + lambda_ * deviation, self.model)

        self.attack_log["lambda"] = float(lambda_)
        self.attack_log["perturb"] = self._perturb_vector

    @property
    def _lambda_range(self) -> Sequence[float]:
        return torch.arange(self._min_lambda, self._max_lambda, step=(self._max_lambda - self._min_lambda) / 100)

    def _perturb(self, ref_update: torch.Tensor, benign_updates: list[torch.Tensor]) -> torch.Tensor:
        return {
            "unit": lambda: - ref_update / ref_update.norm(),
            "std": lambda: - torch.stack(benign_updates).std(dim=0),
            "sign": lambda: - ref_update.sign(),
        }[self._perturb_vector]()

    def _bypass(self, _lambda, deviation, info):
        bad_vector = info.global_vector + (info.reference_update + _lambda * deviation)
        model_utils.vector_to_model(bad_vector, self._buffer)

        models, weight = info.robust_fn(
            global_model=info.global_model,
            local_models=[*info.benign_models, *([self._buffer] * len(info.malicious_clients))],
            weights=[1 / len(info.selected_clients)] * len(info.selected_clients),
        )

        # only krum has flag
        if isinstance(info.robust_fn, robust.Krum):
            if any([hasattr(model, "attack_flag") for model in models]):
                return True
            return False

        # This is slightly different with the original code, since Bulyan use TrimmedMean and can not return
        # selected models. The target is to increase the loss, namely the gradient difference norm.
        if isinstance(info.robust_fn, (robust.TrimmedMean, robust.Median, robust.Bulyan)):
            assert len(models) == 1

            aggregated_update = model_utils.model_to_vector(models[0]) - info.global_vector
            loss = (info.reference_update - aggregated_update).norm()

            if loss > self._prev_loss:
                self._prev_loss = loss
                return True

            return False

        raise NotImplementedError


class MinMaxAttacker(AgrTailoredAttacker):
    def _bypass(self, _lambda, deviation, info):
        bad_update = (info.reference_update + _lambda * deviation)
        distance = max([bad_update.dist(update) ** 2 for update in info.benign_updates])
        max_distance = max([update.dist(other) ** 2 for update in info.benign_updates for other in info.benign_updates])

        if distance <= max_distance:
            self.attack_log["distance"] = float(distance)
            return True
        return False


class MinSumAttacker(AgrTailoredAttacker):
    def _bypass(self, _lambda, deviation, info):
        bad_update = (info.reference_update + _lambda * deviation)
        score = sum([bad_update.dist(update) ** 2 for update in info.benign_updates])
        min_score = min([sum([update.dist(other) ** 2 for other in info.benign_updates])
                         for update in info.benign_updates])

        if score <= min_score:
            self.attack_log["distance"] = float(score)
            return True
        return False
