import random
import torch
from torch.utils.tensorboard import SummaryWriter
from utils_plotting import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from utils_plotting import plot_gate_outputs_to_numpy


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir)

    def log_training(self, loss, grad_norm, lr, duration, iteration):
        self.add_scalar("training/loss", loss[0], iteration)
        self.add_scalar("training/loss_mel_l1", loss[1], iteration)
        self.add_scalar("training/loss_mel_l2", loss[2], iteration)
        self.add_scalar("training/loss_gate", loss[3], iteration)
        self.add_scalar("training/grad_norm", grad_norm, iteration)
        self.add_scalar("training/learning_rate", lr, iteration)
        self.add_scalar("training/duration", duration, iteration)

    def log_validation(self, loss, model, y, y_pred, iteration):
        self.add_scalar("validation/loss", loss[0], iteration)
        self.add_scalar("validation/loss_mel_l1", loss[1], iteration)
        self.add_scalar("validation/loss_mel_l2", loss[2], iteration)
        self.add_scalar("validation/loss_gate", loss[3], iteration)
        _, mel_outputs, gate_outputs, alignments = y_pred
        mel_targets, gate_targets = y

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace(".", "/")
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, gate target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        self.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration,
            dataformats="HWC",
        )
        self.add_image(
            "mel_target",
            plot_spectrogram_to_numpy(mel_targets[idx].data.cpu().numpy()),
            iteration,
            dataformats="HWC",
        )
        self.add_image(
            "mel_predicted",
            plot_spectrogram_to_numpy(mel_outputs[idx].data.cpu().numpy()),
            iteration,
            dataformats="HWC",
        )
        self.add_image(
            "gate",
            plot_gate_outputs_to_numpy(
                gate_targets[idx].data.cpu().numpy(),
                torch.sigmoid(gate_outputs[idx]).data.cpu().numpy(),
            ),
            iteration,
            dataformats="HWC",
        )
