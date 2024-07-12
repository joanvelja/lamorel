import math
import os
import logging
import torch
import numpy as np
from tqdm import tqdm
from lamorel import BaseUpdater
from collections import OrderedDict
from accelerate import Accelerator

accelerator = Accelerator()


class PPOUpdater(BaseUpdater):
    def __init__(
        self,
        model_type,
        minibatch_size,
        gradient_batch_size,
        gradient_minibatch_size=None,
    ):
        super(PPOUpdater, self).__init__()
        self._model_type = model_type
        self._minibatch_size = minibatch_size
        self._gradient_batch_size = gradient_batch_size
        self._gradient_minibatch_size = gradient_minibatch_size

    def _get_trainable_params(self, model, return_with_names=False):
        if return_with_names:
            return filter(lambda p: p[1].requires_grad, model.named_parameters())
        else:
            return filter(lambda p: p.requires_grad, model.parameters())

    def perform_update(self, contexts, candidates, _current_batch_ids, **kwargs):
        if not hasattr(self, "optimizer"):
            self._iterator_named_trainable_params = lambda: self._get_trainable_params(
                self._llm_module, True
            )
            self._iterator_trainable_params = (
                p for n, p in self._iterator_named_trainable_params()
            )
            self.optimizer = torch.optim.Adam(
                self._iterator_trainable_params, lr=kwargs["lr"]
            )

            if os.path.exists(kwargs["loading_path"] + "/optimizer.checkpoint"):
                self.optimizer.load_state_dict(
                    torch.load(kwargs["loading_path"] + "/optimizer.checkpoint")
                )

        current_process_buffer = {}
        for k in ["actions", "advantages", "returns", "logprobs", "values"]:
            current_process_buffer[k] = kwargs[k][_current_batch_ids]

        epochs_losses = {"value": [], "policy": [], "loss": []}

        n_minibatches = math.ceil(len(contexts) / self._minibatch_size)
        for i in tqdm(range(kwargs["ppo_epochs"]), ascii=" " * 9 + ">", ncols=100):
            for step in range(n_minibatches):
                _minibatch_start_idx = step * self._minibatch_size
                _minibatch_end_idx = min(
                    (step + 1) * self._minibatch_size, len(contexts)
                )

                self.optimizer.zero_grad()
                gradient_accumulation_steps = math.ceil(
                    (_minibatch_end_idx - _minibatch_start_idx)
                    / self._gradient_batch_size
                )
                for accumulated_batch in range(gradient_accumulation_steps):
                    _start_idx = (
                        _minibatch_start_idx
                        + accumulated_batch * self._gradient_batch_size
                    )
                    _stop_idx = _minibatch_start_idx + min(
                        (accumulated_batch + 1) * self._gradient_batch_size,
                        _minibatch_end_idx,
                    )

                    _contexts = contexts[_start_idx:_stop_idx]
                    _candidates = candidates[_start_idx:_stop_idx]
                    if len(_contexts) == 0:
                        break
                    if self._gradient_minibatch_size is None:
                        _batch_size = sum(len(_c) for _c in _candidates)
                    else:
                        _batch_size = self._gradient_minibatch_size
                    # Use LLM to compute again action probabilities and value
                    output = self._llm_module(
                        ["score", "value"],
                        contexts=_contexts,
                        candidates=_candidates,
                        require_grad=True,
                        minibatch_size=_batch_size,
                    )
                    scores = torch.stack([_o["score"] for _o in output]).squeeze()
                    probas = torch.distributions.Categorical(logits=scores)
                    values = torch.stack([_o["value"][0] for _o in output]).squeeze()

                    # Compute policy loss
                    entropy = probas.entropy().mean()
                    log_prob = probas.log_prob(
                        current_process_buffer["actions"][_start_idx:_stop_idx]
                    )  # Use logprobs from dist as they were normalized
                    ratio = torch.exp(
                        log_prob
                        - current_process_buffer["logprobs"][_start_idx:_stop_idx]
                    )
                    # assert not (i == 0 and step == 0 and (torch.any(ratio < 0.99) or torch.any(ratio > 1.1)))
                    if (
                        i == 0
                        and step == 0
                        and (torch.any(ratio < 0.99) or torch.any(ratio > 1.1))
                    ):
                        logging.warning("PPO ratio != 1 !!")

                    clip_adv = (
                        torch.clamp(
                            ratio, 1 - kwargs["clip_eps"], 1 + kwargs["clip_eps"]
                        )
                        * current_process_buffer["advantages"][_start_idx:_stop_idx]
                    )
                    policy_loss = -(
                        torch.min(
                            ratio
                            * current_process_buffer["advantages"][
                                _start_idx:_stop_idx
                            ],
                            clip_adv,
                        )
                    ).mean()
                    epochs_losses["policy"].append(policy_loss.detach().cpu().item())

                    # Compute value loss
                    unclipped_value_error = (
                        values - current_process_buffer["returns"][_start_idx:_stop_idx]
                    ) ** 2
                    clipped_values = current_process_buffer["values"][
                        _start_idx:_stop_idx
                    ] + torch.clamp(
                        values - current_process_buffer["values"][_start_idx:_stop_idx],
                        -kwargs["clip_eps"],
                        kwargs["clip_eps"],
                    )
                    clipped_value_error = (
                        clipped_values
                        - current_process_buffer["returns"][_start_idx:_stop_idx]
                    ) ** 2
                    value_loss = torch.max(
                        unclipped_value_error, clipped_value_error
                    ).mean()
                    epochs_losses["value"].append(value_loss.detach().cpu().item())

                    # Compute final loss
                    loss = (
                        policy_loss
                        - kwargs["entropy_coef"] * entropy
                        + kwargs["value_loss_coef"] * value_loss
                    )
                    loss = loss / gradient_accumulation_steps
                    epochs_losses["loss"].append(loss.detach().cpu().item())

                    # Backward
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._iterator_trainable_params, kwargs["max_grad_norm"]
                )
                self.optimizer.step()

        if kwargs["save_after_update"] and accelerator.process_index == 1:
            print("Saving model...")
            model_state_dict = OrderedDict(
                {k: v for k, v in self._iterator_named_trainable_params()}
            )
            torch.save(model_state_dict, kwargs["output_dir"] + "/model.checkpoint")
            torch.save(
                self.optimizer.state_dict(),
                kwargs["output_dir"] + "/optimizer.checkpoint",
            )
            print("Model saved")

        return {
            "loss": np.mean(epochs_losses["loss"]),
            "value_loss": np.mean(epochs_losses["value"]),
            "policy_loss": np.mean(epochs_losses["policy"]),
        }
