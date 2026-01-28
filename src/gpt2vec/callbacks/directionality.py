import numpy as np
import torch
from typing import Dict, Any, Iterable

from transformers import TrainerCallback
from transformers.integrations import WandbCallback
from gpt2vec.metrics.directionality import (
    AttentionExtractor,
    SymmetryScore,
    DirectionalityScore,
)


class AttentionGeometryCallback(WandbCallback):
    def __init__(
        self,
        q_path: str,
        k_path: str,
        layers: Iterable[int] | int | None = None,
        is_lora: bool = False,
        merge_lora: bool = False,
        adapter_name: str | None = None,
        is_quantized: bool = False,
        attention_type: str | None = None,
    ):
        super().__init__()
        self.q_path = q_path
        self.k_path = k_path
        self.layers = [layers] if isinstance(layers, int) else layers

        self.extractor_kwargs = dict(
            attention_type=attention_type,
            is_lora=is_lora,
            merge_lora=merge_lora,
            adapter_name=adapter_name,
            is_quantized=is_quantized,
        )


    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:          # HF passes an empty dict sometimes
            return

        # run only on rank-0 to avoid duplicate logs under DDP / DS
        if getattr(args, "local_rank", -1) not in (-1, 0):
            return

        model = kwargs["model"]

        with torch.no_grad():     # absolute safety: never touch graph
            extractor = AttentionExtractor(
                model,
                self.q_path,
                self.k_path,
                **self.extractor_kwargs,
            )
            sym  = SymmetryScore(extractor)(self.layers)
            dirc = DirectionalityScore(extractor)(self.layers)


        # HF Trainer will push whatever is inside `logs`
        if self.layers is None:
            self.layers = range(len(sym))

        name = "attn"
        if self.extractor_kwargs['merge_lora'] is False:
            name = "lora_delta"

        tmp_logs = {}
        for layer in self.layers:
            tmp_logs[f"{name}/symmetry/layer_{layer}"] = sym[layer]
            tmp_logs[f"{name}/directionality/layer_{layer}"] = dirc[layer]

        tmp_logs[f"{name}/avg_symmetry"] = np.array(sym).mean()
        tmp_logs[f"{name}/avg_directionality"] = np.array(dirc).mean()

        self._wandb.log(tmp_logs, step=state.global_step)