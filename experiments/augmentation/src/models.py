import torch
import torch.nn as nn
from transformers import AutoModel


class FlatHateModel(nn.Module):
    """Flat multi-task classifier.

    coarse: 3-class single-label (clean / offensive / hate)
    fine: 8-way multi-label classification
    loss: CE(coarse) + lambda_fine * BCE(fine)
    """

    def __init__(
        self,
        plm_name: str,
        num_coarse: int,
        num_fine: int,
        lambda_fine: float = 1.0,
        class_weight_coarse: torch.Tensor = None,
        pos_weight_fine: torch.Tensor = None,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(plm_name)
        hidden_size = self.encoder.config.hidden_size

        self.head_coarse = nn.Linear(hidden_size, num_coarse)
        self.head_fine = nn.Linear(hidden_size, num_fine)
        self.lambda_fine = lambda_fine

        self.crit_coarse = nn.CrossEntropyLoss(weight=class_weight_coarse)
        self.crit_fine = nn.BCEWithLogitsLoss(pos_weight=pos_weight_fine)

    def forward(self, input_ids, attention_mask, label_coarse=None, label_fine=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h_cls = out.last_hidden_state[:, 0]

        logits_coarse = self.head_coarse(h_cls)
        logits_fine = self.head_fine(h_cls)

        result = {
            "logits_coarse": logits_coarse,
            "logits_fine": logits_fine,
        }

        if label_coarse is not None and label_fine is not None:
            label_coarse = label_coarse.view(-1).long()
            label_fine = label_fine.float()
            if label_fine.dim() == 1:
                label_fine = label_fine.view(-1, 1)

            loss_coarse = self.crit_coarse(logits_coarse, label_coarse)
            loss_fine = self.crit_fine(logits_fine, label_fine)
            result["loss"] = loss_coarse + self.lambda_fine * loss_fine
            result["loss_coarse"] = loss_coarse
            result["loss_fine"] = loss_fine

        return result


class HierHateModel(nn.Module):
    """Hierarchical classifier with differentiable coarse/fine consistency loss.

    coarse: 0=clean, 1=offensive, 2=hate
    fine: 8-way multi-label classification
    loss: CE + lambda_fine * BCE + lambda_hier * consistency loss

    The consistency term penalizes:
    - high clean probability together with high probability of any fine hate label
    - high toxic probability together with high probability of no fine hate label
    """

    def __init__(
        self,
        plm_name: str,
        num_coarse: int = 3,
        num_fine: int = 8,
        lambda_fine: float = 1.0,
        lambda_hier: float = 1.0,
        class_weight_coarse: torch.Tensor = None,
        pos_weight_fine: torch.Tensor = None,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(plm_name)
        hidden_size = self.encoder.config.hidden_size

        self.head_coarse = nn.Linear(hidden_size, num_coarse)
        self.head_fine = nn.Linear(hidden_size, num_fine)
        self.lambda_fine = lambda_fine
        self.lambda_hier = lambda_hier

        self.crit_coarse = nn.CrossEntropyLoss(weight=class_weight_coarse)
        self.crit_fine = nn.BCEWithLogitsLoss(pos_weight=pos_weight_fine)

    def forward(self, input_ids, attention_mask, label_coarse=None, label_fine=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h_cls = out.last_hidden_state[:, 0]

        logits_coarse = self.head_coarse(h_cls)
        logits_fine = self.head_fine(h_cls)

        outputs = {
            "logits_coarse": logits_coarse,
            "logits_fine": logits_fine,
        }

        if label_coarse is not None and label_fine is not None:
            label_coarse = label_coarse.view(-1).long()
            label_fine = label_fine.float()
            if label_fine.dim() == 1:
                label_fine = label_fine.view(-1, 1)

            loss_coarse = self.crit_coarse(logits_coarse, label_coarse)
            loss_fine = self.crit_fine(logits_fine, label_fine)

            coarse_prob = torch.softmax(logits_coarse, dim=-1)
            fine_prob = torch.sigmoid(logits_fine)

            clean_prob = coarse_prob[:, 0]
            toxic_prob = coarse_prob[:, 1:].sum(dim=1)
            no_fine_prob = torch.prod(1.0 - fine_prob, dim=1)
            any_fine_prob = 1.0 - no_fine_prob

            clean_with_fine = clean_prob * any_fine_prob
            toxic_without_fine = toxic_prob * no_fine_prob
            hier_penalty = (clean_with_fine + toxic_without_fine).mean()

            loss = (
                loss_coarse
                + self.lambda_fine * loss_fine
                + self.lambda_hier * hier_penalty
            )

            outputs["loss"] = loss
            outputs["loss_coarse"] = loss_coarse
            outputs["loss_fine"] = loss_fine
            outputs["loss_hier"] = hier_penalty

        return outputs
