from typing import Optional, Union
import copy

import torch
from torch import nn

from transformers.utils import logging
from transformers.cache_utils import Cache, StaticCache, HybridCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    MaskedLMOutput,
    TokenClassifierOutput
)
from transformers.processing_utils import Unpack

from transformers import Gemma3PreTrainedModel, Gemma3TextModel, Gemma3TextConfig, Gemma3ForCausalLM
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3TextScaledWordEmbedding,
    Gemma3DecoderLayer,
    Gemma3Attention,
    Gemma3MLP,
    Gemma3RMSNorm,
    Gemma3RotaryEmbedding, # TODO: try to customize this
)

from transformers.utils import (
    is_torch_flex_attn_available,
    logging,
)

from gpt2vec.losses.losses import ForMaskedLMLoss, ForMaskedNTPLoss


logger = logging.get_logger(__name__)



class biGemma3Attention(Gemma3Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False


class biGemma3DecoderLayer(Gemma3DecoderLayer):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = biGemma3Attention(config=config, layer_idx=layer_idx)

        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.is_sliding = self.self_attn.is_sliding
        self.sliding_window = config.sliding_window


class biGemma3TextModel(Gemma3TextModel):
    def __init__(self, config: Gemma3TextConfig):
        Gemma3PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Gemma3 downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5. See https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5
        )
        self.layers = nn.ModuleList(
            [biGemma3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # TODO: raushan fix this after RoPE refactor. For now we hack it by reassigning thetas
        # when we want to create a local RoPE layer. Config defaults should hold values for global RoPE
        config = copy.deepcopy(config)
        config.rope_theta = config.rope_local_base_freq
        config.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = Gemma3RotaryEmbedding(config=config)

        # Initialize weights and apply final processing
        self.post_init()


    @torch.no_grad()
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: HybridCache,
        output_attentions: bool,
    ):
        # Flash Attention currently doesn't support static cache but Gemma3Text work only with static cache.
        # So we will pass in attention mask as is in any case, not only when ther's padding. Then we'll use its shape
        # to cut out keys/values trailing 0 used in static cache. This workaround should be compile compatible
        # as it doesn't cause dynamic control issues.
        if self.config._attn_implementation == "flash_attention_2":
            return attention_mask

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if isinstance(past_key_values, (HybridCache, StaticCache)):
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = attention_mask.shape[-1] if attention_mask is not None else input_tensor.shape[1]

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )
        return causal_mask


    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        min_dtype = torch.finfo(dtype).min

        # in original implementation:
        # causal_mask = torch.full(
        #     (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
        # )
        causal_mask = torch.zeros(
            (sequence_length, target_length), dtype=dtype, device=cache_position.device
        )

        # Commenting out next 2 lines to disable causal masking
        # if sequence_length != 1:
        #     causal_mask = torch.triu(causal_mask, diagonal=1)

        causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone() # copy to contiguous memory for in-place edit

            # original implementation for 2D attention mask
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
            
            # custom implementation for 4D attention mask
            elif attention_mask.dim() == 4:
                # backwards compatibility: we allow passing a 4D attention mask shorter than the input length with
                # cache. In that case, the 4D attention mask attends to the newest tokens only.
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0],
                    : mask_shape[1],
                    offset : mask_shape[2] + offset,
                    : mask_shape[3],
                ] = mask_slice

        return causal_mask


# initialized with a random head
class biGemma3ForMaskedLM(Gemma3PreTrainedModel):
    config_class = Gemma3TextConfig
    base_model_prefix = "language_model"

    def __init__(self, config):
        super().__init__(config)
        self.model = biGemma3TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> MaskedLMOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = ForMaskedLMLoss(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
     


# with trained head
class biGemma3ForMaskedNTP(Gemma3ForCausalLM):
    def __init__(self, config):
        Gemma3PreTrainedModel.__init__(self, config)
        self.model = biGemma3TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = ForMaskedNTPLoss(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class biGemma3ForTokenClassification(Gemma3PreTrainedModel):
    config_class = Gemma3TextConfig
    base_model_prefix = "language_model"

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = biGemma3TextModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs
    ) -> TokenClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )