import re
from typing import Iterable, List, Optional, Sequence, Union

import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


class PartialGradNormCallback(TrainerCallback):
    def __init__(
        self,
        model: torch.nn.Module,
        target_modules: Union[str, Sequence[str], None] = None,
        target_layers: Optional[Iterable[int]] = None,
        norm_type: float = 2.0,
        log_key: str = "partial_grad_norm",
    ):
        super().__init__()
        self.model = model
        if target_modules is None:
            target_modules = []
        if isinstance(target_modules, str):
            target_modules = [target_modules]

        self._regexes: List[re.Pattern] = [re.compile(p) for p in target_modules]
        self._target_layers = (
            set(int(i) for i in target_layers) if target_layers is not None else None
        )
        self.norm_type = norm_type
        self.log_key = log_key
        self._cached_norm: Optional[float] = None


    def _is_selected(self, param_name: str) -> bool:
        if self._regexes:  # empty list -> keep everything
            if not any(r.search(param_name) for r in self._regexes):
                return False

        if self._target_layers is not None:
            m = re.search(r"\.layer[s]?\.([0-9]+)\.", param_name)
            if m is None or int(m.group(1)) not in self._target_layers:
                return False
            
        return True
    
    def _compute_grad_norm(self) -> float:
        norms = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            if self._is_selected(name):
                norms.append(p.grad.detach().norm(self.norm_type))

        if len(norms) == 0:
            return 0.0

        stacked = torch.stack(norms)
        return stacked.norm(self.norm_type).item()


    # ----------------------- Trainer hooks ----------------------- #

    def on_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Grads exist at this point (before optimizer.step / zero_grad)
        if control.should_log:
            self._cached_norm = self._compute_grad_norm()
        else:
            self._cached_norm = None

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        if self._cached_norm is not None and logs is not None:
            logs[self.log_key] = self._cached_norm
