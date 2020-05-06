import random
import torch
from torch.utils.tensorboard import SummaryWriter
from plotting_utils import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from plotting_utils import plot_gate_outputs_to_numpy


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(
        self, reduced_loss, grad_norm, learning_rate, duration, iteration
    ):
        self.add_scalar("training.loss", reduced_loss, iteration)
        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)
        self.add_scalar("duration", duration, iteration)

    def log_validation(self, reduced_loss, model, y, y_pred, iteration):
        self.add_scalar("validation.loss", reduced_loss, iteration)
        [outs_left, outs_right] = y_pred
        mel_outs_l, gate_outs_l, aligns_l, _ = outs_left
        _, mel_outs_r, gate_outs_r, aligns_r, _, aligns_aux = outs_right

        mel_targets, gate_targets = y
        mel_targets_left = mel_targets['left']
        gate_targets_left = gate_targets['left']
        mel_targets_right = mel_targets['right']
        gate_targets_right = gate_targets['right']

        # plot distribution of parameters
        # for tag, value in model.named_parameters():
        #     tag = tag.replace(".", "/")
        #     self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, mel_outs_l.size(0) - 1)
        self.add_image(
            "alignment for left decoder",
            plot_alignment_to_numpy(aligns_l[idx].data.cpu().numpy().T),
            iteration,
            dataformats="HWC",
        )
        self.add_image(
            "alignment for right decoder",
            plot_alignment_to_numpy(aligns_r[idx].data.cpu().numpy().T),
            iteration,
            dataformats="HWC",
        )
        self.add_image(
            "alignment for auxiliary information",
            plot_alignment_to_numpy(aligns_aux[idx].data.cpu().numpy().T),
            iteration,
            dataformats="HWC",
        )

        self.add_image(
            "mel_target for left decoder",
            plot_spectrogram_to_numpy(mel_targets_left[idx].data.cpu().numpy()),
            iteration,
            dataformats="HWC",
        )
        self.add_image(
            "mel_predicted for left decoder",
            plot_spectrogram_to_numpy(mel_outs_l[idx].data.cpu().numpy()),
            iteration,
            dataformats="HWC",
        )
        self.add_image(
            "mel_target for right decoder",
            plot_spectrogram_to_numpy(mel_targets_right[idx].data.cpu().numpy()),
            iteration,
            dataformats="HWC",
        )
        self.add_image(
            "mel_predicted for right decoder",
            plot_spectrogram_to_numpy(mel_outs_r[idx].data.cpu().numpy()),
            iteration,
            dataformats="HWC",
        )

        self.add_image(
            "gate for left decoder",
            plot_gate_outputs_to_numpy(
                gate_targets_left[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outs_l[idx]).data.cpu().numpy(),
            ),
            iteration,
            dataformats="HWC",
        )
        self.add_image(
            "gate for right decoder",
            plot_gate_outputs_to_numpy(
                gate_targets_right[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outs_r[idx]).data.cpu().numpy(),
            ),
            iteration,
            dataformats="HWC",
        )

    def log_assist(self, string_assist, value_assist, iteration):
        self.add_scalar(string_assist, value_assist, iteration)
