"""
Back to the Drawing Board: A Critical Evaluation of Poisoning Attacks on Production Federated Learning.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.optim as optim

import flcore
import flcore.utils.model as model_utils
from attackers.core import Attacker, AttackInfo

__all__ = ["PgaAttacker"]


class PgaAttacker(Attacker):
    _buffer: nn.Module = None

    def __init__(self, *, client: flcore.Client, max_scale: float, **kwargs):
        super().__init__(client=client, **kwargs)
        self._max_scale = max_scale

    def attack_algorithm(self, info: AttackInfo):
        if self._buffer is None:
            self.__class__._buffer = copy.deepcopy(self.model)

        # Forget learned model
        self.receive_model(info.global_model)

        # (Algo 1, line 2) an average of the norms of some benign updates available to her
        tao = torch.stack(info.benign_updates).norm(dim=1).mean(dim=0)

        # (Algo 1, line 3) Stochastic Gradient Ascent
        self._sga()

        # (Algo 1, line 4)
        bad_update = model_utils.model_to_vector(self.model) - info.global_vector

        # (Algo 1, line 5) (Algo 2)
        bad_update = self._project(bad_update, tao, info)

        # Return
        model_utils.vector_to_model(info.global_vector + bad_update, self.model)

    def _sga(self):
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)

        for local_epoch in range(self.max_epoch):
            for x, y in self.train_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                y_hat = self.model(x)
                loss = self.loss_fn(y_hat, y)

                # NOTE: This is a slight difference from the paper, but it is necessary to avoid nan
                if loss > 5000:
                    loss = torch.log(loss)
                (-loss).backward()  # SGA by negativing loss

                self.attack_log["loss"] = float(loss)

                optimizer.step()

        # incase nan
        vector = model_utils.model_to_vector(self.model)
        vector = torch.nan_to_num(vector)
        model_utils.vector_to_model(vector, self.model)

    def _project(self, bad_update: torch.Tensor, tao: float, info: AttackInfo):
        # (Algo 2, line 2) d*
        max_deviation = 0

        # (Algo 2, line 3) gamma*
        max_gamma = 1

        # (Algo 2, line 4) Scale bad_grad to have norm tao
        bad_update = bad_update * tao / bad_update.norm()

        # (Algo 2, line 5) # Compute reference benign gradient / update
        ref_update = info.reference_update

        # (Algo 2, line 6-11)
        for gamma in torch.arange(1, self._max_scale, (self._max_scale - 1) / 100):  # try 100 times for simplicity
            bad_update_2 = gamma * bad_update

            # line 8
            model_utils.vector_to_model(info.global_vector + bad_update_2, self._buffer)
            models, weights = info.robust_fn(
                global_model=info.global_model,
                local_models=[*info.benign_models, *([self._buffer] * len(info.malicious_clients))],
                weights=[1 / len(info.selected_clients)] * len(info.selected_clients)
            )
            aggregated_vector = model_utils.aggregate_vector(info.global_vector,
                                                             [model_utils.model_to_vector(model) for model in models],
                                                             weights or [1 / len(models)] * len(models))
            aggregated_update = aggregated_vector - info.global_vector
            deviation = (aggregated_update - ref_update).norm()

            if deviation > max_deviation:
                max_gamma = gamma
                max_deviation = deviation

        self.attack_log["gamma"] = float(max_gamma)
        self.attack_log["deviation"] = float(max_deviation)

        return max_gamma * bad_update
