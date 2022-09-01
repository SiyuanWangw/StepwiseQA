from transformers import AutoModel, PreTrainedModel
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch
import torch.nn.functional as F


class IntSuppFactModel(PreTrainedModel):

    def __init__(self,
                 config,
                 args
                 ):
        super().__init__(config)
        self.args = args

        self.encoder = AutoModel.from_pretrained(self.args.model_name)

        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.sp = nn.Linear(config.hidden_size, 1)

    def forward(self, batch):
        # first hop
        outputs = self.encoder(batch['q_sp_input_ids'], batch['q_sp_mask'], batch.get('q_sp_type_ids', None))
        sequence_output = outputs[0]
        
        sent_mask = (batch["sent_offsets"] != 0).long()
        gather_index = batch["sent_offsets"].unsqueeze(2).expand(-1, -1, sequence_output.size()[-1])
        sent_marker_rep = torch.gather(sequence_output, 1, gather_index)
        sp_score = self.sp(sent_marker_rep).squeeze(2) - 1e30 * (1 - sent_mask)

        # second hop
        outputs_2 = self.encoder(batch['q_sp_input_ids_2'], batch['q_sp_mask_2'], batch.get('q_sp_type_ids_2', None))
        sequence_output_2 = outputs_2[0]

        logits = self.qa_outputs(sequence_output_2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        sent_mask_2 = (batch["sent_offsets_2"] != 0).long()
        gather_index_2 = batch["sent_offsets_2"].unsqueeze(2).expand(-1, -1, sequence_output_2.size()[-1])
        sent_marker_rep_2 = torch.gather(sequence_output_2, 1, gather_index_2)
        sp_score_2 = self.sp(sent_marker_rep_2).squeeze(2) - 1e30 * (1 - sent_mask_2)

        if self.training:
            sp_loss = F.binary_cross_entropy_with_logits(sp_score, batch["first_sent_labels"].float(), reduction="none")
            sp_loss = sp_loss * sent_mask

            sp_loss = sp_loss.sum() / sent_mask.sum()

            sp_loss_2 = F.binary_cross_entropy_with_logits(sp_score_2, batch["second_sent_labels"].float(), reduction="none")
            sp_loss_2 = sp_loss_2 * sent_mask_2

            sp_loss_2 = sp_loss_2.sum() / sent_mask_2.sum()

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
            span_loss = start_loss + end_loss

            return sp_loss, sp_loss_2, span_loss

        return {
            'start_logits': start_logits,
            'end_logits': end_logits,
            "sp_scores": sp_score,
            "sp_scores_2": sp_score_2,
            }
