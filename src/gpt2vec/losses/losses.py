import torch.nn as nn
from transformers.loss.loss_utils import fixed_cross_entropy, ForMaskedLMLoss

# Loss is similar to ForMaskedLMLoss, but we need to shift the labels
def ForMaskedNTPLoss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: int = None,
    ignore_index: int = -100,
    shift_labels=None,
    **kwargs,
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = labels.to(logits.device)

    if shift_labels is None:
        labels = labels.to(logits.device)
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss


LOSS_MAPPING = {
    "mlm": ForMaskedLMLoss,
    "mntp": ForMaskedNTPLoss
}