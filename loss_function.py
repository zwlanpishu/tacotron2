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
        pos_weight = gate_out.new_full((1,), 6.0)

        # Custom defined weights for each loss
        mel_loss_l1 = nn.L1Loss()(mel_out, mel_target) + nn.L1Loss()(
            mel_out_postnet, mel_target
        )
        mel_loss_l2 = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(
            mel_out_postnet, mel_target
        )
        gate_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(gate_out, gate_target)
        return (
            mel_loss_l1 + mel_loss_l2 + gate_loss,
            (mel_loss_l1, mel_loss_l2, gate_loss),
        )
