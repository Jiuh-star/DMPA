"""
Durable Model Poisoning Attack
"""
from __future__ import annotations

import copy
import itertools
from typing import Literal, TYPE_CHECKING

import torch
import torch.nn as nn

import flcore.utils.model as model_utils

from attackers.core import Attacker, AttackInfo


if TYPE_CHECKING:
    from system.fedavg import FedAvgClient

__all__ = ["DmpaAttacker"]


class DmpaAttacker(Attacker):
    _clients: list[FedAvgClient] = []
    _bad_model: nn.Module = None
    _attack_at: int = -1
    # _state_dict: dict = {}
    _buffer: nn.Module = None

    def __init__(
        self,
        *,
        client: FedAvgClient,
        learning_rate: float,
        max_epoch: int,
        momentum: float,
        max_scale: float,
        method: Literal["norm", "param", "param norm", "norm param"],
        max_loss: float,
        reverse_factor: float,
        skip_too_large_loss: bool,
        **kwargs,
    ):
        super().__init__(client=client, **kwargs)
        self._lr = learning_rate
        self._max_epoch = max_epoch
        self._momentum = momentum
        self._max_scale = max_scale
        self._method = method
        self._max_loss = max_loss
        self._reverse_factor = reverse_factor
        self._skip_too_large_loss = skip_too_large_loss
        self._norm_supremum = -1
        self._norm_infimum = -1
        self.__class__._clients.append(client)
        client.connect()  # Keep client alive

    def attack_algorithm(self, info: AttackInfo):
        # To speed up attack algorithm
        if self._attack_at == info.global_epoch:
            model_utils.move_parameters(self._bad_model.to(self.device), self.model)
            self.attack_log["attacked"] = True
            return

        if self._buffer is None:
            self.__class__._buffer = copy.deepcopy(self.model)

        # Forget learned model
        self.receive_model(info.global_model)

        # Get an optimizer for attacking
        optimizer_type = torch.optim.SGD
        optimizer = optimizer_type(self.model.parameters(), lr=self._lr, momentum=self._momentum, maximize=True)

        # Momentum between attacks
        # if self._state_dict:
        #     optimizer.load_state_dict(self._state_dict)

        for i in range(self._max_epoch):
            losses = []
            self._reset_constrain_param(info)

            for x, y in self._data_iter():
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                y_hat = self.model(x)
                loss = self.loss_fn(y_hat, y)
                
                # Training time will be unbearable when loss too large
                if self._skip_too_large_loss and loss > 10 * self._max_loss:
                    continue

                # Incase some of the losses growing up too fast
                if loss > self._max_loss:
                    print("Constrain Loss:", loss.item())
                    loss = torch.log(loss)

                # Gradient ascend
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                # Constraint malicious momentum update
                vector = model_utils.model_to_vector(self.model)
                update = vector - info.global_vector

                if i > self._max_epoch / 2:
                    update = self._infimum_constrain(info, update)  # Infimum constrain to enhance bad update performance

                # reversed weight decay
                # if self._method == "param":
                #     update = update + 0.005 * vector / vector.norm()
                update = self._supremum_constrain(info, update)  # Supremum to make sure final malicious update update can bypass robust algo

                vector = info.global_vector + update
                model_utils.vector_to_model(vector, self.model)

            # For analyze
            if losses:
                self.attack_log["loss"] = sum(losses) / len(losses)
                self.attack_log["loss (min)"] = min(losses)
                self.attack_log["loss (max)"] = max(losses)

            print(
                "Epoch:",
                i,
                "Loss:",
                abs(self.attack_log["loss"]),
                "Loss (min):",
                abs(self.attack_log["loss (min)"]),
                "Loss (max):",
                abs(self.attack_log["loss (max)"]),
                "Norm:",
                (model_utils.model_to_vector(self.model) - info.global_vector).norm().item(),
                "Scale:",
                self.attack_log.get("scale_factor", -1),
                "Deviation:",
                self.attack_log.get("deviation", -1),
                "Infimum:",
                self._norm_infimum,
                "Supremum:",
                self._norm_supremum,
            )

        # Build final malicious model
        self._reset_constrain_param(info)

        vector = model_utils.model_to_vector(self.model)
        update = vector - info.global_vector
        update = update - self._reverse_factor * info.reference_update  # better than positive version
        # update = self._infimum_constrain(info, update)
        update = self._supremum_constrain(info, update)

        # mitigrate poison
        # update = 0.5 * update

        vector = info.global_vector + update
        model_utils.vector_to_model(vector, self.model)

        self.__class__._attack_at = info.global_epoch
        self.__class__._bad_model = copy.deepcopy(self.model)
        # self.__class__._state_dict = optimizer.state_dict()

        self.attack_log["lr"] = self._lr
        self.attack_log["epoch"] = self._max_epoch
        self.attack_log["max_scale"] = self._max_scale

    # NOTE: The code will be available after the paper is received.