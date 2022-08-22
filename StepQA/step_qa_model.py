from transformers import AutoModel, PreTrainedModel
import torch.nn as nn
from torch.nn import CrossEntropyLoss, NLLLoss
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

class StepQAModel(PreTrainedModel):

    def __init__(self,
                 config,
                 args
                 ):
        super().__init__(config)
        self.args = args

        self.encoder = AutoModel.from_pretrained(self.args.model_name)

        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.sp = nn.Linear(config.hidden_size, 1)

        self.end = nn.Linear(config.hidden_size, 1)
        if "electra" in args.model_name:
            self.pooler = BertPooler(config)

    def encode_inter_sp(self, sp_input, sent_offset):
        outputs = self.encoder(sp_input['input_ids'], sp_input['attention_mask'], sp_input.get('token_type_ids', None))
        sequence_output = outputs[0]

        # first_pool_output = self.pooler(sequence_output)
        # first_end_score = self.end(first_pool_output)

        sent_mask = (sent_offset != 0).long()
        gather_index = sent_offset.unsqueeze(2).expand(-1, -1, sequence_output.size()[-1])
        sent_marker_rep = torch.gather(sequence_output, 1, gather_index)
        sp_score = self.sp(sent_marker_rep).squeeze(2) - 1e30 * (1 - sent_mask)

        return sp_score

    def encode_last(self, sp_input, sent_offset):
        outputs = self.encoder(sp_input['input_ids'], sp_input['attention_mask'], sp_input.get('token_type_ids', None))
        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        sent_mask = (sent_offset != 0).long()
        gather_index = sent_offset.unsqueeze(2).expand(-1, -1, sequence_output.size()[-1])
        sent_marker_rep = torch.gather(sequence_output, 1, gather_index)
        sp_score = self.sp(sent_marker_rep).squeeze(2) - 1e30 * (1 - sent_mask)

        return start_logits, end_logits, sp_score

    def forward(self, batch):
        # first hop
        outputs = self.encoder(batch['q_sp_input_ids'], batch['q_sp_mask'], batch.get('q_sp_type_ids', None))
        sequence_output = outputs[0]
        
        first_pool_output = self.pooler(sequence_output)
        first_end_score = self.end(first_pool_output)
        
        sent_mask = (batch["sent_offsets"] != 0).long()
        gather_index = batch["sent_offsets"].unsqueeze(2).expand(-1, -1, sequence_output.size()[-1])
        sent_marker_rep = torch.gather(sequence_output, 1, gather_index)
        sp_score = self.sp(sent_marker_rep).squeeze(2) - 1e30 * (1 - sent_mask)

        # second hop
        outputs_2 = self.encoder(batch['q_sp_input_ids_2'], batch['q_sp_mask_2'], batch.get('q_sp_type_ids_2', None))
        sequence_output_2 = outputs_2[0]
        
        second_end_score = self.end(self.pooler(sequence_output_2))

        sent_mask_2 = (batch["sent_offsets_2"] != 0).long()
        gather_index_2 = batch["sent_offsets_2"].unsqueeze(2).expand(-1, -1, sequence_output_2.size()[-1])
        sent_marker_rep_2 = torch.gather(sequence_output_2, 1, gather_index_2)
        sp_score_2 = self.sp(sent_marker_rep_2).squeeze(2) - 1e30 * (1 - sent_mask_2)

        # third hop
        outputs_3 = self.encoder(batch['q_sp_input_ids_3'], batch['q_sp_mask_3'], batch.get('q_sp_type_ids_3', None))
        sequence_output_3 = outputs_3[0]

        third_end_score = self.end(self.pooler(sequence_output_3))
        
        sent_mask_3 = (batch["sent_offsets_3"] != 0).long()
        gather_index_3 = batch["sent_offsets_3"].unsqueeze(2).expand(-1, -1, sequence_output_3.size()[-1])
        sent_marker_rep_3 = torch.gather(sequence_output_3, 1, gather_index_3)
        sp_score_3 = self.sp(sent_marker_rep_3).squeeze(2) - 1e30 * (1 - sent_mask_3)

        # forth hop
        outputs_4 = self.encoder(batch['q_sp_input_ids_4'], batch['q_sp_mask_4'], batch.get('q_sp_type_ids_4', None))
        sequence_output_4 = outputs_4[0]

        logits = self.qa_outputs(sequence_output_4)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        sent_mask_4 = (batch["sent_offsets_4"] != 0).long()
        gather_index_4 = batch["sent_offsets_4"].unsqueeze(2).expand(-1, -1, sequence_output_4.size()[-1])
        sent_marker_rep_4 = torch.gather(sequence_output_4, 1, gather_index_4)
        sp_score_4 = self.sp(sent_marker_rep_4).squeeze(2) - 1e30 * (1 - sent_mask_4)

        if self.training:
            sp_loss = F.binary_cross_entropy_with_logits(sp_score, batch["first_sent_labels"].float(), reduction="none")
            sp_loss = sp_loss * sent_mask

            sp_loss = sp_loss.sum() / sent_mask.sum()

            sp_loss_2 = F.binary_cross_entropy_with_logits(sp_score_2, batch["second_sent_labels"].float(), reduction="none")
            sp_loss_2 = sp_loss_2 * sent_mask_2

            sp_loss_2 = sp_loss_2.sum() / sent_mask_2.sum()
            
            sp_loss_3 = F.binary_cross_entropy_with_logits(sp_score_3, batch["third_sent_labels"].float(), reduction="none")
            sp_loss_3 = sp_loss_3 * sent_mask_3
            
            sp_loss_3 = sp_loss_3.sum() / sent_mask_3.sum()


            # intermediate sf prediction loss
            # masked average
            inter_sp_loss = sp_loss + (sp_loss_2 + sp_loss_3) * (batch["first_ends"] == 0).long()
            inter_sp_loss = inter_sp_loss / (1+2*(batch["first_ends"] == 0).long())
            inter_sp_loss = inter_sp_loss.mean()

            # # average
            # inter_sp_loss = (sp_loss + sp_loss_2 + sp_loss_3) / 3

            sp_loss_4 = F.binary_cross_entropy_with_logits(sp_score_4, batch["forth_sent_labels"].float(), reduction="none")
            sp_loss_4 = sp_loss_4 * sent_mask_4

            sp_loss_4 = sp_loss_4.sum() / sent_mask_4.sum()

            end_loss_mask_1 = (batch["first_ends"] >= 0).long()
            end_loss_mask_2 = (batch["second_ends"] >= 0).long()
            end_loss_mask_3 = (batch["third_ends"] >= 0).long() 

            first_end_target = batch["first_ends"]
            first_end_target[first_end_target<0] = 0
            end_loss_1 = F.binary_cross_entropy_with_logits(first_end_score, first_end_target.float(), reduction="none")

            second_end_target = batch["second_ends"]
            second_end_target[second_end_target<0] = 0
            end_loss_2 = F.binary_cross_entropy_with_logits(second_end_score, second_end_target.float(), reduction="none")

            third_end_target = batch["third_ends"]
            third_end_target[third_end_target<0] = 0
            end_loss_3 = F.binary_cross_entropy_with_logits(third_end_score, third_end_target.float(), reduction="none")

            # end prediction loss
            reason_end_loss = end_loss_1 * end_loss_mask_1 + end_loss_2 * end_loss_mask_2 + end_loss_3 * end_loss_mask_3
            reason_end_loss = reason_end_loss / (end_loss_mask_1 + end_loss_mask_2 + end_loss_mask_3)
            reason_end_loss = reason_end_loss.mean()

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

            return inter_sp_loss, sp_loss_4, span_loss, reason_end_loss

        return {
            'start_logits': start_logits,
            'end_logits': end_logits,
            "sp_scores": sp_score,
            "sp_scores_2": sp_score_2,
            "sp_scores_3": sp_score_3,
            "sp_scores_4": sp_score_4,
            "end_score_1": first_end_score,
            "end_score_2": second_end_score,
            "end_score_3": third_end_score
            }
