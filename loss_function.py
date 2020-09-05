from torch import nn


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        pos_weight = gate_out.new_full((1,), 1.0)
        mel_loss = 0.5 * nn.MSELoss()(
            mel_out, mel_target
        ) + 0.5 * nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(
            gate_out, gate_target
        )
        return mel_loss + gate_loss
