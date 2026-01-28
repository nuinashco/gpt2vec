import torch
from typing import Tuple, List


class UTERScore:
    """
    Computes the positional Upper Triangle Energy Ratio (pUTER) score for attention layers.
    """
    def __init__(self):
        self._buffer: List[List[float]] = []

    def add_batch(self,
                  attentions: Tuple[torch.Tensor, ...],
                  attn_mask:   torch.Tensor):
        self._buffer.append(self._uter_per_layer(attentions, attn_mask))


    def compute(self) -> List[float]:
        if not self._buffer:
            return []
        stack = torch.tensor(self._buffer)         # (micro_batches, L)
        return stack.mean(dim=0).tolist()          # layer means


    def reset(self):
        self._buffer.clear()


    @staticmethod
    def _uter_per_layer(
            attentions: Tuple[torch.Tensor, ...],
            attn_mask: torch.Tensor
    ) -> List[float]:
        mask_q = attn_mask.unsqueeze(1).unsqueeze(-1)
        mask_k = attn_mask.unsqueeze(1).unsqueeze(-2)
        valid  = (mask_q & mask_k).to(attentions[0].dtype)

        eps, uters = 1e-9, []
        for A in attentions:
            tri_u = torch.triu(A, diagonal=1)
            tri_valid = torch.triu(valid, diagonal=1)

            num   = ((tri_u * tri_valid) ** 2).sum(dim=(-1, -2))
            denom = ((A * valid) ** 2).sum(dim=(-1, -2))
            uters.append((num / (denom + eps)).mean().item())
        return uters