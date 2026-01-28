import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from typing import Dict, Any
from gpt2vec.metrics.uter import UTERScore


class UTERCallback(TrainerCallback):
    def __init__(self, device: str | None = None):
        self.device = device
        self.metric = UTERScore()

    def on_train_batch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Dict[str, Any]
    ):
        inputs = kwargs["inputs"]
        outputs = kwargs["outputs"]
        model = kwargs["model"]
        device  = self.device or next(model.parameters()).device

        if isinstance(outputs, dict) and "attentions" in outputs:
            attentions = outputs["attentions"]
        else:
            # run a quick no-grad forward pass to obtain attentions
            with torch.no_grad():
                re_out = model(**{k: v.to(device) for k, v in inputs.items()},
                               output_attentions=True,
                               return_dict=True)
            attentions = re_out.attentions

        attn_mask = inputs["attention_mask"].to(device)
        self.metric.add_batch(attentions, attn_mask)


    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        layer_means = self.metric.compute()
        if layer_means:
            logs = {f"uter/L{i}": v for i, v in enumerate(layer_means)}
            logs["uter/mean"] = sum(layer_means) / len(layer_means)
            self.log(logs)

        self.metric.reset()