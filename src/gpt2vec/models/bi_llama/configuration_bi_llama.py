from transformers.models.llama import LlamaConfig


class biLlamaConfig(LlamaConfig):
    model_type = "bi_llama"

    def __init__(
        self,
        use_bidirectional_attention=False,
        **kwargs
    ):
        self.use_bidirectional_attention = use_bidirectional_attention
        super().__init__(**kwargs)


__all__ = [
    "biLlamaConfig"
]