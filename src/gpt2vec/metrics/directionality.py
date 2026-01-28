import torch
import functools

import bitsandbytes as bnb
from bitsandbytes.functional import dequantize_4bit

from abc import ABC, abstractmethod
from typing import Literal, Dict, Any, Optional, List, Iterable



def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


class AttentionExtractor:
    def __init__(
        self,
        model,
        q_path: str,
        k_path: str,
        attention_type: Optional[Literal["grouped"]] = None,

        is_lora: bool = False,
        merge_lora: bool = False,
        adapter_name: Optional[str] = None,

        is_quantized: bool = False,
    ):
        self.model = model
        self.q_path, self.k_path = q_path, k_path
        self.attention_type = attention_type

        self.is_lora = is_lora
        self.merge_lora = merge_lora
        self.adapter_name = adapter_name  # default: merge *all* loaded adapters

        self.is_quantized = is_quantized

        self.layer_count = model.config.num_hidden_layers
        self.d  = model.config.hidden_size
        self.dh = self.d // model.config.num_attention_heads

    # def layer_count(self) -> int:
    #     prefix, _ = self.q_path.split("[layer_idx].")
    #     return len(rgetattr(self.model, prefix))


    def _get_weight(self, lin: torch.nn.Linear | bnb.nn.Linear4bit) -> torch.Tensor:
        """
        Get the weights of a linear layer, handling quantization if necessary.
        """
        if self.is_quantized:
            assert isinstance(lin, bnb.nn.Linear4bit), \
                f"Expected Linear4bit, got {type(lin)}"

            packed = lin.weight.data
            qs = lin.weight.quant_state
            weight = dequantize_4bit(packed, qs)

        else:
            assert isinstance(lin, torch.nn.Linear), \
                f"Expected Linear, got {type(lin)}"

            weight = lin.weight

        return weight


    def matrix(self, layer_idx: int) -> torch.Tensor:
        """Return Q·Kᵀ for the chosen layer as a *detached* tensor."""
        Wq = self._raw(self.q_path, layer_idx).T.detach()
        Wk = self._raw(self.k_path, layer_idx).T.detach()

        if self.attention_type == "grouped":
            # "Grouped" attention (Mistral-like) -> reshape keys before mm
            Wk = Wk.view(Wk.shape[0], self.dh, Wk.shape[1] // self.dh)
            rep = (Wq.shape[0] // self.dh) // Wk.shape[-1]
            Wk = Wk.repeat_interleave(rep, 0).view(Wq.shape[0], Wq.shape[0])

        return Wq @ Wk.T


    def _raw(self, path: str, idx: int) -> torch.Tensor:
        """
        path: str
            Path to the matrix in the model, e.g. "model.layers[layer_idx].self_attn.q_proj".
            The [layer_idx] part will be replaced with the actual layer index.
        """

        layers_path, matrix_path = path.split("[layer_idx].")
        layer_module = rgetattr(self.model, layers_path)[idx]
        proj_module = rgetattr(layer_module, matrix_path)

        if not self.is_lora:
            assert not self._is_lora_layer(proj_module), \
                f"Module {proj_module} is a LoRA layer, but is_lora is False."

            return self._get_weight(proj_module)

        else:
            delta_lora = self._delta_lora_weight(proj_module)

            if self.merge_lora:
                return self._get_weight(proj_module.base_layer) + delta_lora
            else:
                return delta_lora


    @staticmethod
    def _is_lora_layer(module) -> bool:
        try:
            from peft.tuners.lora import LoraLayer
            return isinstance(module, LoraLayer)
        except ImportError:
            return hasattr(module, "lora_A") and hasattr(module, "lora_B")


    def _delta_lora_weight(self, module) -> torch.Tensor:
        delta = torch.zeros_like(self._get_weight(module.base_layer))

        if not self._is_lora_layer(module):
            return delta

        adapters = (
            [self.adapter_name] if self.adapter_name is not None
            else module.lora_A.keys()
        )
        for name in adapters:
            A = module.lora_A[name].weight # [r, in]
            B = module.lora_B[name].weight # [out, r]
            r = A.size(0)
            alpha = (
                module.lora_alpha[name]
                if isinstance(module.lora_alpha, dict)
                else module.lora_alpha
            )
            delta += (B @ A) * (alpha / r)

        return delta

        

class AttentionScorer(ABC):
    def __init__(self, extractor: AttentionExtractor):
        self.extractor = extractor

    def __call__(self, layers: Iterable[int] | int | None = None) -> List[float] | float:
        if layers is None:
            layers = range(self.extractor.layer_count)
        if isinstance(layers, int):
            layers = [layers]

        with torch.no_grad():
            scores = [self._score(self.extractor.matrix(i)) for i in layers]
        return scores if len(scores) > 1 else scores[0]

    @abstractmethod
    def _score(self, A: torch.Tensor) -> float: ...


class SymmetryScore(AttentionScorer):
    def _score(self, A: torch.Tensor) -> float:
        sym = 0.5 * (A + A.T)

        # somehow we should handle the case when A is zero
        denominator = (A ** 2).sum()
        if denominator == 0:
            return 1.0
            
        score = (sym ** 2).sum() / denominator

        # like in paper
        score = 2 * score - 1

        return score.detach().item()


class DirectionalityScore(AttentionScorer):
    def _score(self, A: torch.Tensor, *, num_std: int = 2) -> float:
        row, col = torch.norm(A, dim=1), torch.norm(A, dim=0)
        rt, ct = row.mean() + num_std*row.std(), col.mean() + num_std*col.std()
        r_exc, c_exc = torch.sum(row[row > rt] - rt), torch.sum(col[col > ct] - ct)
        total = r_exc + c_exc
        score = 0.0 if total == 0 else (c_exc - r_exc) / total

        # like in paper
        score = -1 * score

        return score.detach().item()