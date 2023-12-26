"""
X. Cao and N. Z. Gong,
“MPAF: Model Poisoning Attacks to Federated Learning based on Fake Clients,”
in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), Jun. 2022, pp.
3395–3403.
doi: 10.1109/CVPRW56347.2022.00383.
"""
from __future__ import annotations

import torch.nn as nn

import flcore
import flcore.utils.model as model_utils
from attackers.core import Attacker, AttackInfo

__all__ = ["MpafAttacker"]


class MpafAttacker(Attacker):
    def __init__(self, *, client: flcore.Client, base_model: nn.Module, lambda_: float, max_norm: float, **kwargs):
        super().__init__(client=client, **kwargs)
        self._lambda = lambda_
        self._max_norm = max_norm

        self._base_vector = model_utils.model_to_vector(base_model).to(self.device)

    def attack_algorithm(self, info: AttackInfo):
        # malicious_model = global_model + lambda * (base_model - global_model)
        # so in the server:
        # global_model = global_model + weight * (malicious_model - global_model) + ε
        # ideally
        # global_model = base_model + ε
        global_vector = info.global_vector

        malicious_vector = global_vector + self._lambda * (self._base_vector - global_vector)

        # for fair comparison durability under the almost same norm
        # update = malicious_vector - global_vector
        # if 0 < self._max_norm < update.norm().item():
        #     malicious_vector = global_vector + (update / update.norm()) * self._max_norm

        model_utils.vector_to_model(malicious_vector, self.model)

        self.attack_log["lambda"] = float(self._lambda)
