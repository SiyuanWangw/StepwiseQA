from transformers import AutoModel, PreTrainedModel
import torch.nn as nn
import torch
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


class ParaSelectModel(PreTrainedModel):

    def __init__(self,
                 config,
                 args
                 ):
        super().__init__(config)
        self.args = args

        self.encoder = AutoModel.from_pretrained(self.args.model_name)

        if "electra" in args.model_name:
            self.pooler = BertPooler(config)

        self.sp = nn.Linear(config.hidden_size, 1)

    def forward(self, batch):
        outputs = self.encoder(batch['q_sp_input_ids'], batch['q_sp_mask'], batch.get('q_sp_type_ids', None))
        sequence_output = outputs[0]

        sent_mask = (batch["sent_offsets"] != 0).long()
        gather_index = batch["sent_offsets"].unsqueeze(2).expand(-1, -1, sequence_output.size()[-1])
        sent_marker_rep = torch.gather(sequence_output, 1, gather_index)
        sp_score = self.sp(sent_marker_rep).squeeze(2) - 1e30 * (1 - sent_mask)

        if self.training:
            sp_loss = F.binary_cross_entropy_with_logits(sp_score, batch["para_labels"].float(), reduction="none")
            sp_loss = sp_loss * sent_mask
            sp_loss = sp_loss.sum() / sent_mask.sum()

            return sp_loss

        return {
            "sp_scores": sp_score
            }