"""
Denial-of-Service or Fine-Grained Control: Towards Flexible Model Poisoning Attacks on Federated Learning
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Iterable

import torch
import torch.nn as nn
import torch.optim as optim

import flcore.utils.model as model_utils
from .core import Attacker, AttackInfo

if TYPE_CHECKING:
    from system.fedavg import FedAvgClient

__all__ = ["FmpaAttacker"]


class PerturbedGradientDescent(optim.Optimizer):
    """
    This optimizer should be able to solve the optimization problem as mentioned at Equation 5.
    Note that this optimizer is common to solve the federated learning problem under heterogeneous data distribution.
    """
    def __init__(self, params: Iterable[torch.Tensor] | Iterable[dict], lr: float, lambda_: float):
        assert lr > 0.0
        assert lambda_ > 0.0
        defaults = {"lr": lr, "lambda": lambda_}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, global_params: Iterable[torch.Tensor]):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                if p.grad is None:
                    continue

                # Ok for negative loss, which gradient ascent while keep gradient small
                dp = p.grad.data + group['lambda'] * (p.data - g.data)
                p.data.add_(dp, alpha=-group['lr'])


class FmpaAttacker(Attacker):
    """
    This is a partial implementation of the I-FMPA, which removed finetune, just for comparison.
    """
    _clients: list[FedAvgClient] = []
    _attack_at: int = -1
    _bad_model: nn.Module = None

    def __init__(self, *, client: FedAvgClient, max_scale: int, perturb_lambda: float, **kwargs):
        super().__init__(client=client, **kwargs)

        self._max_scale = max_scale
        self._perturb_lambda = perturb_lambda

        self.__class__._clients.append(client)
        client.connect()

    def attack_algorithm(self, info: AttackInfo):
        if self._attack_at == info.global_epoch:
            model_utils.move_parameters(self._bad_model.to(self.device), self.model)
            return

        # Forget learned model
        self.receive_model(info.global_model)

        # NOTE: In original implementation on https://github.com/ZhangHangTao/Poisoning-Attack-on-FL, which is submitted
        # delete action from author, the attacker should prepare a malicious model ahead-of-time from a pickle file.
        # However, no generation process is provided in its implementation. We have no choice but to train a bad model
        # from scratch using the SOTA poisoning method we know to generate a poisoned model. As a result some experiment
        # results may be different from the original paper.

        min_dis = 1_000_000
        decay = 1.1
        scale = self._max_scale

        # Firstly, generating a malicious model that satisfy Equation 5.
        # This optimizer
        optimizer = PerturbedGradientDescent(self.model.parameters(), lr=0.01, lambda_=self._perturb_lambda)

        self.model.train()
        for x, y in self._data_iter():
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            y_hat = self.model(x)
            loss = self.loss_fn(y_hat, y)

            # in case nan
            if loss > 100_000:
                loss = torch.log(loss)

            asc_loss = - loss  # gradient ascent by negating the loss
            asc_loss.backward()
            optimizer.step(info.global_model.parameters())

        # Secondly, find certified radius and project it on ball
        # NOTE: Here we follow the method described in author's implementation, which certified radius may not much
        # clear in code.

        buffer_model = copy.deepcopy(self.model)
        target_vector = model_utils.model_to_vector(self.model)
        update = model_utils.model_to_vector(self.model) - info.global_vector

        while True:
            scaled_vector = info.global_vector + scale * update
            model_utils.vector_to_model(scaled_vector, buffer_model)
            local_models = list(info.benign_models) + ([buffer_model] * len(info.malicious_clients))
            models, weights = info.robust_fn(
                global_model=info.global_model,
                local_models=local_models,
                weights=[1 / len(local_models)] * len(local_models),
            )
            agg_vector = model_utils.aggregate_vector(
                info.global_vector,
                [model_utils.model_to_vector(model) for model in models],
                weights
            )

            # We choose L2 loss, one of the author's implementations.
            now_dis = (agg_vector - target_vector).norm(p=2)

            if now_dis <= min_dis:
                min_dis = now_dis
                scale /= decay
            else:
                scale *= decay
                break

        # Finally, generating the malicious model and update.
        model_utils.vector_to_model(scaled_vector, self.model)

        self.__class__._attack_at = info.global_epoch
        self.__class__._bad_model = copy.deepcopy(self.model)

        self.attack_log["scale"] = float(scale)
        self.attack_log["radius"] = float(min_dis)

    def _data_iter(self):
        for client in self._clients:
            for x, y in client.train_dataloader:
                yield x, y
