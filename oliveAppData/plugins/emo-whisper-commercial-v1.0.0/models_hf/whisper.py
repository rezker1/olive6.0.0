# models/ssl_gap.py
#
# Author: Ewald Enzinger (SRI International)
# Date: 2024-03-29
#
# Model using Wav2Vec2 pretrained model + global average pooling (avg+std) + 2x dense/linear

from typing import Dict, Tuple, Union, Optional
import torch
from transformers import AutoModelForAudioXVector, AutoConfig
from transformers.models.whisper.modeling_whisper import (
    WhisperPreTrainedModel,
    WhisperEncoder,
    WhisperConfig,
)
from transformers.modeling_outputs import XVectorOutput


class WhisperEncoderAvgPoolConfig(WhisperConfig):
    model_type = "WhisperEncoderAvgPool"


class WhisperEncoderAvgPool(WhisperPreTrainedModel):
    config_class = WhisperEncoderAvgPoolConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config):
        super().__init__(config)

        self.encoder = WhisperEncoder(config)

        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        print(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_features: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], XVectorOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_features,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        hidden_states = encoder_outputs[0]
        output_embeddings = hidden_states.mean(dim=1)

        logits = self.classifier(output_embeddings)

        loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # move labels to correct device to enable PP
            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + encoder_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return XVectorOutput(
            loss=loss,
            logits=logits,
            embeddings=output_embeddings,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

