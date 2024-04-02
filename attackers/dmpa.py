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

    def _reset_constrain_param(self, info: AttackInfo):
        if "param" in self._method:
            self._reset_param_constrain_param(info)
        if "norm" in self._method:
            self._reset_norm_constrain_param(info)

    def _infimum_constrain(self, info: AttackInfo, update: torch.Tensor):
        if self._method == "param":
            update = self._param_infimum_constrain(info, update)
        elif self._method == "norm":
            update = self._norm_infimum_constrain(info, update)
        elif self._method == "param norm":
            update = self._param_infimum_constrain(info, update)
            update = self._norm_infimum_constrain(info, update)
        elif self._method == "norm param":
            update = self._norm_infimum_constrain(info, update)
            update = self._param_infimum_constrain(info, update)
        else:
            raise ValueError(f"Unknown method {self.method}")
        return update

    def _supremum_constrain(self, info: AttackInfo, update: torch.Tensor):
        if self._method == "param":
            update = self._param_supremum_constrain(info, update)
        elif self._method == "norm":
            update = self._norm_supremum_constrain(info, update)
        elif self._method == "param norm":
            update = self._param_supremum_constrain(info, update)
            update = self._norm_supremum_constrain(info, update)
        elif self._method == "norm param":
            update = self._norm_supremum_constrain(info, update)
            update = self._param_supremum_constrain(info, update)
        else:
            raise ValueError(f"Unknown method {self.method}")

        return update

    def _reset_param_constrain_param(self, info: AttackInfo, **kwargs):
        self._last_param_update = None
        benign_updates = torch.stack(info.benign_updates)
        self._supremum_max_boundary = benign_updates.max(dim=0)[0]
        self._supremum_min_boundary = benign_updates.min(dim=0)[0]

        # wrong implementation
        # self._infimum_max_boundary = self._supremum_max_boundary
        # self._infimum_min_boundary = self._supremum_min_boundary
        
        # scale supremum? bad idea
        # mid = self._supremum_max_boundary + self._supremum_min_boundary
        # self._supremum_max_boundary = 1.5 * (mid - self._supremum_max_boundary) + mid
        # self._supremum_min_boundary = 1.5 * (mid - self._supremum_min_boundary) + mid

        # mid = (self._supremum_max_boundary + self._supremum_min_boundary) / 2
        # self._infimum_max_boundary = (mid + self._supremum_max_boundary) / 2
        # self._infimum_min_boundary = (mid + self._supremum_min_boundary) / 2
        
        # 9 / 10
        # self._infimum_max_boundary = (9 * self._supremum_max_boundary + self._supremum_min_boundary) / 10
        # self._infimum_min_boundary = (9 * self._supremum_min_boundary + self._supremum_min_boundary) / 10

        # negative number is better?
        # mid = (self._supremum_max_boundary + self._supremum_min_boundary) / 2
        # self._infimum_max_boundary = self._supremum_max_boundary
        # self._infimum_min_boundary = (mid + self._supremum_min_boundary) / 2

        # positive number is better?
        # mid = (self._supremum_max_boundary + self._supremum_min_boundary) / 2
        # self._infimum_max_boundary = (mid + self._supremum_max_boundary) / 2
        # self._infimum_min_boundary = self._supremum_min_boundary

        # prefer larger norm of the global model? larger norm, but same durability
        # mid = (self._supremum_max_boundary + self._supremum_min_boundary) / 2
        # vector = model_utils.model_to_vector(self.model)
        # self._infimum_max_boundary = (mid + self._supremum_max_boundary) / 2
        # self._infimum_min_boundary = (mid + self._supremum_min_boundary) / 2
        # mask = vector > 0
        # self._infimum_max_boundary[mask] = ((mid + self._supremum_max_boundary) / 2)[mask]
        # self._infimum_min_boundary[mask] = self._supremum_min_boundary[mask]
        # mask = vector < 0
        # self._infimum_max_boundary[mask] = self._supremum_max_boundary[mask]
        # self._infimum_min_boundary[mask] = ((mid + self._supremum_min_boundary) / 2)[mask]

        # stay away from ref_update
        ref_update = info.reference_update
        self._infimum_max_boundary = (1 * self._supremum_max_boundary + ref_update) / 2
        self._infimum_min_boundary = (1 * self._supremum_min_boundary + ref_update) / 2

        # stay away from ref_update, but consider trimmed boundary? not that bad
        # m = len(info.malicious_clients)
        # ref_updates = torch.stack(info.benign_updates).sort(dim=0)[0][m // 2: -m // 2]
        # ref_update = ref_updates.mean(dim=0)[0]
        # self._supremum_max_boundary = ref_updates.max(dim=0)[0] * 2
        # self._supremum_min_boundary = ref_updates.min(dim=0)[0] * 2
        # self._infimum_max_boundary = (1 * self._supremum_max_boundary + ref_update) / 2
        # self._infimum_min_boundary = (1 * self._supremum_min_boundary + ref_update) / 2

        # stay away from ref_update, but prefer larger norm of the global model?
        # m = len(info.malicious_clients)
        # ref_updates = torch.stack(info.benign_updates).sort(dim=0)[0][m // 2: -m // 2]
        # ref_update = ref_updates.mean(dim=0)[0]
        # self._supremum_max_boundary = ref_updates.max(dim=0)[0]
        # self._supremum_min_boundary = ref_updates.min(dim=0)[0]
        # mid = (self._supremum_max_boundary + self._supremum_min_boundary) / 2
        # mid = ref_update
        # vector = info.global_vector
        # self._infimum_max_boundary = (mid + self._supremum_max_boundary) / 2
        # self._infimum_min_boundary = (mid + self._supremum_min_boundary) / 2
        # mask = vector > 0
        # self._infimum_max_boundary[mask] = ((mid + self._supremum_max_boundary) / 2)[mask]
        # self._infimum_min_boundary[mask] = self._supremum_min_boundary[mask]
        # mask = vector < 0
        # self._infimum_max_boundary[mask] = self._supremum_max_boundary[mask]
        # self._infimum_min_boundary[mask] = ((mid + self._supremum_min_boundary) / 2)[mask]

        # unit?
        # from .lmpa import bisect_right
        # lambda_ = self._max_scale
        # self._prev_loss = -1
        # update = model_utils.model_to_vector(self.model) - info.global_vector
        # deviation = update / update.norm()
        # self._lambda_range = torch.arange(1E-3, self._max_scale, step=(self._max_scale - 1E-3) / 100)
        # index = bisect_right(self._lambda_range, bypass=lambda x: self._bypass(x, deviation, info))
        # self._unit_lambda = self._lambda_range[index]

    def _param_infimum_constrain(self, info: AttackInfo, update: torch.Tensor):
        if self._last_param_update is None:
            self._last_param_update = copy.deepcopy(update)
            return update

        direction = update - self._last_param_update
        self._last_param_update = update

        block = torch.logical_and(update > self._infimum_min_boundary, update < self._infimum_max_boundary)
        mask = torch.logical_and(direction > 0, block)
        update[mask] = self._infimum_max_boundary[mask]
        mask = torch.logical_and(direction < 0, block)
        update[mask] = self._infimum_min_boundary[mask]

        # just go that direction?
        # update = self._max_scale * update

        # unit?
        # update = self._unit_lambda * (update / update.norm())

        # try to scale? no
        # update = update / update.norm()
        # buffer = copy.deepcopy(self.model)
        # prev_loss = 0
        # max_lambda = 1
        # for lambda_ in torch.arange(1, self._max_scale, 0.1):
        #     bad_vector = info.global_vector + lambda_ * update
        #     model_utils.vector_to_model(bad_vector, buffer)
        #     models, weight = info.robust_fn(
        #         global_model=info.global_model,
        #         local_models=[*info.benign_models, *([buffer] * len(info.malicious_clients))],
        #         weights=[1 / len(info.selected_clients)] * len(info.selected_clients),
        #     )
        #     aggregated_update = model_utils.model_to_vector(models[0]) - info.global_vector
        #     loss = (info.reference_update - aggregated_update).norm()
        #     if loss > prev_loss:
        #         prev_loss = loss
        #         max_lambda = lambda_
        # 
        # update = max_lambda * update

        return update

    def _param_supremum_constrain(self, info: AttackInfo, update: torch.Tensor):
        mask = update > self._supremum_max_boundary
        update[mask] = self._supremum_max_boundary[mask]

        mask = update < self._supremum_min_boundary
        update[mask] = self._supremum_min_boundary[mask]

        return update

    def _reset_norm_constrain_param(self, info: AttackInfo):
        self._norm_infimum = -1
        self._norm_supremum = -1

    def _norm_supremum_constrain(self, info: AttackInfo, update: torch.Tensor):
        norm_infimum, norm_supremum = self._find_norm_robust_boundary(info, update)
        update_norm = update.norm().item()

        # Incase Infimum > Supremum
        if norm_infimum > norm_supremum:
            norm_supremum = norm_infimum

        # Supremum
        if update_norm > norm_supremum:
            update = update / update_norm * norm_supremum

        return update

    def _norm_infimum_constrain(self, info: AttackInfo, update: torch.Tensor):
        norm_infimum, norm_supremum = self._find_norm_robust_boundary(info, update)
        update_norm = update.norm().item()

        # Infimum
        if update_norm < norm_infimum:
            update = update / update_norm * norm_infimum

        return update

    def _find_norm_robust_boundary(self, info: AttackInfo, update: torch.Tensor):
        if self._norm_infimum > 0 and self._norm_supremum > 0:
            return self._norm_infimum, self._norm_supremum

        # We don't have to calculate boundary every time, change lr to a smaller value may be better and quicker
        buffer = copy.deepcopy(self.model)
        update = update / update.norm() * torch.stack(info.benign_updates).norm(dim=1).mean()
        max_deviation = -1
        max_scale_factor = 1

        # Find the robust boundary
        for scale_factor in itertools.chain(
            torch.arange(0.001, 1, (1 - 0.001) / 50),
            torch.arange(1, self._max_scale, (self._max_scale - 1) / 100),
        ):
            # If there are no robust function, we can do whatever we want
            if not info.robust_fn:
                max_scale_factor = self._max_scale
                break

            scaled_vector = info.global_vector + scale_factor * update

            # evaluate this scaled model can or cannot be aggregated into global model
            model_utils.vector_to_model(scaled_vector, buffer)
            local_models = list(info.benign_models) + ([buffer] * len(info.malicious_clients))
            robust_models, weights = info.robust_fn(
                global_model=info.global_model,
                local_models=local_models,
                # attack should not know the weights of benign clients
                weights=[1 / len(local_models)] * len(local_models),
            )
            agg_vector = model_utils.aggregate_vector(
                info.global_vector,
                [model_utils.model_to_vector(model) for model in robust_models],
                weights,
            )
            agg_update = agg_vector - info.global_vector
            deviation = (agg_update.dist(info.reference_update) ** 2).item()
            # We aim to move model to step away from final global model (~=reference_model)
            # deviation = (agg_vector.dist(info.reference_vector) ** 2).item()

            # if this scaled update cause a larger deviation than before, remember it
            if deviation > max_deviation:
                max_deviation = deviation
                max_scale_factor = scale_factor.item()

        # After finding the robust boundary, we know the infimum and supremum
        self._norm_supremum = (max_scale_factor * update).norm().item()
        self._norm_infimum = max(info.reference_update.norm().item(), self._norm_supremum * 0.75)
        self._scale_factor = max_scale_factor

        self.attack_log["scale_factor"] = max_scale_factor
        self.attack_log["deviation"] = max_deviation
        self.attack_log["infimum"] = self._norm_infimum
        self.attack_log["supremum"] = self._norm_supremum

        return self._norm_infimum, self._norm_supremum

    def _data_iter(self):
        for client in self._clients:
            for x, y in client.train_dataloader:
                yield x, y


DualProjectionAttacker = Ours2Attacker

