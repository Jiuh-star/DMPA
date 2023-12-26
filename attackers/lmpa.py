"""
M. Fang, X. Cao, J. Jia, and N. Gong,
“Local Model Poisoning Attacks to Byzantine-Robust Federated Learning,”
in 29th USENIX Security Symposium (USENIX Security 20), 2020, pp.
1605–1622. [Online].
Available: https://www.usenix.org/conference/usenixsecurity20/presentation/fang
"""
from __future__ import annotations

import copy
import math
from typing import Sequence

import torch
import torch.nn as nn

import flcore
import flcore.utils.model as model_utils
import flcore.utils.robust as robust
from attackers.core import Attacker, AttackInfo

__all__ = ["LmpaAttacker", "bisect_right"]


class LmpaAttacker(Attacker):
    _last_attack: int = -1
    _buffer: nn.Module = None

    def __init__(self, *, client: flcore.Client, max_scale: float | int, **kwargs):
        super().__init__(client=client, **kwargs)

        self._max_lambda = max_scale

    def attack_algorithm(self, info: AttackInfo):
        if self._buffer is None:
            bad_model = copy.deepcopy(self.model)
            setattr(bad_model, "bad_flag", True)
            self.__class__._buffer = bad_model

        if isinstance(info.robust_fn, robust.Krum):
            self._try_to_attack(info)
        elif isinstance(info.robust_fn, (robust.TrimmedMean, robust.Median)):
            self._influence_attack(info)
        else:
            raise NotImplementedError

    @staticmethod
    def _get_lambda_range(info: AttackInfo) -> Sequence[float | torch.Tensor]:
        threshold = 1E-5  # From paper

        # Theorem 1
        m = len(info.selected_clients)
        c = len(info.malicious_clients)
        d = info.global_vector.numel()

        # term 1
        distances = [torch.stack([vector.dist(other) for other in info.benign_vectors])
                     for vector in info.benign_vectors]
        min_term = min([dist.sort()[0][1: m - c - 2 + 1].sum() for dist in distances])  # the smallest is itself
        term_1 = min_term / ((m - 2 * c - 1) * math.sqrt(d))

        # term 2
        w_re = info.global_vector
        term_2 = max([vector.dist(w_re) for vector in info.benign_vectors]) / math.sqrt(d)

        upper_bound = float(term_1 + term_2)

        step = (upper_bound - threshold) / 100
        if upper_bound > threshold and step > 0:
            return torch.arange(threshold, upper_bound, step=step)

        return [threshold]

    def _try_to_attack(self, info: AttackInfo):
        def bypass(_lambda):
            bad_vector = info.global_vector - _lambda * direction
            model_utils.vector_to_model(bad_vector, self._buffer)

            models, weights = info.robust_fn(
                info.global_model,
                [*info.benign_models, *([self._buffer] * len(info.malicious_clients))],
                weights=[1 / len(info.selected_clients)] * len(info.selected_clients)
            )

            # only for Krum
            if any([hasattr(model, "attack_flag") for model in models]):
                return True
            return False

        direction = info.reference_update.sign()
        lambda_range = self._get_lambda_range(info)
        index = bisect_right(lambda_range, bypass)
        lambda_ = lambda_range[index]

        model_utils.vector_to_model(info.global_vector - lambda_ * direction, self.model)

        # log
        self.attack_log["lambda"] = float(lambda_)

    def _influence_attack(self, info: AttackInfo):
        b = 2  # A kind of lambda, from paper

        direction = info.reference_update.sign()
        benign_vectors = torch.stack(info.benign_vectors)

        w_max = benign_vectors.max(dim=0)[0]
        w_min = benign_vectors.min(dim=0)[0]

        bad_vector = model_utils.model_to_vector(self.model).zero_()

        # for s_j = -1 and w_max,j > 0, sample from [w_max,j, b * w_max,j]
        mask = (direction < 0) & (w_max > 0)
        sample = torch.rand(bad_vector.size()).to(direction.device)
        sample = (2 * w_max - w_max) * sample + w_max
        sample = sample.masked_select(mask)
        bad_vector.masked_scatter_(mask, sample)

        # for s_j = -1 and w_max,j <= 0, sample from [w_max,j, w_max,j / b]
        mask = (direction < 0) & (w_max <= 0)
        sample = torch.rand(bad_vector.size()).to(direction.device)
        sample = - ((w_max / b - w_max) * sample + w_max)
        sample = sample.masked_select(mask)
        bad_vector.masked_scatter_(mask, sample)

        # for s_j = 1 and w_min,j > 0, sample from [w_min,j / b, w_min,j]
        mask = (direction > 0) & (w_min > 0)
        sample = torch.rand(bad_vector.size()).to(direction.device)
        sample = (w_min - w_min / b) * sample + (w_min / b)
        sample = sample.masked_select(mask)
        bad_vector.masked_scatter_(mask, sample)

        # for s_j = 1 and w_min,j <= 0, sample from [b * w_min, w_min]
        mask = (direction > 0) & (w_min <= 0)
        sample = torch.rand(bad_vector.size()).to(direction.device)
        sample = - ((w_min - b * w_min) * sample + (b * w_min))
        sample = sample.masked_select(mask)
        bad_vector.masked_scatter_(mask, sample)

        model_utils.vector_to_model(bad_vector, self.model)


def bisect_right(a, bypass):
    lo, hi = 0, len(a) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if bypass(a[mid]):  # if succeeded, try larger lambda
            lo = mid + 1
        else:
            hi = mid  # else try smaller lambda

    return lo
