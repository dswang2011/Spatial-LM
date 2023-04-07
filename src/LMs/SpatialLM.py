import torch
import torch.nn as nn
from transformers.modeling_outputs import MaskedLMOutput
from typing import List, Optional, Tuple, Union
from transformers import AutoConfig, AutoModel, LayoutLMv3Model, AutoModelForTokenClassification, AutoModelForQuestionAnswering
from transformers.activations import gelu  # ACT2FN
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_utils import PreTrainedModel


class SpatialLMPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = AutoConfig
    base_model_prefix = "spatial_lm"

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class SpatialLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        # For accelerate compatibility and to not break backward compatibility
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias


class SpatialLMForMaskedLM(SpatialLMPreTrainedModel):
    def __init__(self, config):
        super(SpatialLMForMaskedLM, self).__init__(config)
        self.spatial_lm = LayoutLMv3Model(config, has_relative_attention_bias = False)
        self.lm_head = SpatialLMHead(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        pixel_values: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:

        outputs = self.spatial_lm(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
        )

        # get [B, S, D] (not pooled), and predict complete sequence
        # sequence_output = outputs[0]
        # We need to remove the vision part
        input_shape = input_ids.size()
        seq_length = input_shape[1]
        # only take the text part of the output representations, i.e., [B, S-text, D]
        sequence_output = outputs[0][:, :seq_length]

        prediction_scores = self.lm_head(sequence_output)

        # calculate the loss btw predicted sequence and true sequence
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1),
            )
        else:
            print('==== why there is no sequencial labels?=======')

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return outputs


class SpatialLMForTokenclassifier(SpatialLMPreTrainedModel):
    def __init__(self, opt, freeze_bert=False):
        super(SpatialLMForTokenclassifier, self).__init__()
        self.opt = opt
        # self.config = AutoConfig.from_pretrained(opt.spatial_lm_dir, num_labels=xxx)
        # self.spatial_lm = LayoutLMv3Model(config=self.config)
        self.spatial_lm_token_classifier = AutoModelForTokenClassification.from_pretrained(opt.spatial_lm_dir, num_labels=opt.num_labels,
                                                                        label2id=opt.label2id, id2label=opt.id2label)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        pixel_values: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:

        outputs = self.spatial_lm_token_classifier(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
        )

        return outputs

class SpatialLMForDocVQA(SpatialLMPreTrainedModel):
    def __init__(self, opt, freeze_bert=False):
        super(SpatialLMForDocVQA, self).__init__()
        self.opt = opt
        # self.config = AutoConfig.from_pretrained(opt.spatial_lm_dir, num_labels=xxx)
        # self.spatial_lm = LayoutLMv3Model(config=self.config)
        self.spatial_lm_token_classifier = AutoModelForQuestionAnswering.from_pretrained(opt.spatial_lm_dir)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        pixel_values: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:

        outputs = self.spatial_lm_token_classifier(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
            start_positions = start_positions,
            end_positions = end_positions
        )

        return outputs


if __name__=='__main__':
    spatial_mlm = SpatialLMForMaskedLM(None)
    print(spatial_mlm)

