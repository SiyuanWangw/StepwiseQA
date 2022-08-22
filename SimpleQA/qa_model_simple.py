from transformers import AutoModel, PreTrainedModel
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class SimpleQAModel(PreTrainedModel):

    def __init__(self,
                 config,
                 args
                 ):
        super().__init__(config)
        self.args = args

        self.encoder = AutoModel.from_pretrained(self.args.model_name)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

    def encode_simple(self, sp_input):
        outputs = self.encoder(sp_input['input_ids'], sp_input['attention_mask'], sp_input.get('token_type_ids', None))
        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        return start_logits, end_logits

    def forward(self, batch):
        outputs = self.encoder(batch['q_sp_input_ids'], batch['q_sp_mask'], batch.get('q_sp_type_ids', None))
        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        if self.training:
            start_positions, end_positions = batch["starts"], batch["ends"]

            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduction="mean")
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            span_loss = (start_loss + end_loss) / 2

            return span_loss

        return {
            'start_logits': start_logits,
            'end_logits': end_logits,
            }