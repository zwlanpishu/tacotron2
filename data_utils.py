import random
import numpy as np
import torch
import torch.utils.data

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """

    def __init__(self, audiopaths_and_text, hparams, shuffle=True):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.stft = layers.TacotronSTFT(
            hparams.filter_length,
            hparams.hop_length,
            hparams.win_length,
            hparams.n_mel_channels,
            hparams.sampling_rate,
            hparams.mel_fmin,
            hparams.mel_fmax,
        )
        random.seed(hparams.seed)
        if shuffle:
            random.shuffle(self.audiopaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return (text, mel)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError(
                    "{} {} SR doesn't match target {} SR".format(
                        sampling_rate, self.stft.sampling_rate
                    )
                )
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(
                audio_norm, requires_grad=False
            )
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert (
                melspec.size(0) == self.stft.n_mel_channels
            ), "Mel dimension mismatch: given {}, expected {}".format(
                melspec.size(0), self.stft.n_mel_channels
            )

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate:
    """ Zero-pads model inputs and targets based on number of frames per setep
    """

    def __init__(self, n_frames_left, n_frames_right):
        self.n_frames_left = n_frames_left
        self.n_frames_right = n_frames_right

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0,
            descending=True,
        )
        max_input_len = input_lengths[0]
        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, : text.size(0)] = text

        # Right zero-pad mel-spec with extra single zero vector to mark the end
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch]) + 1

        # prepare data for the left decoder and right decoder respectively
        if max_target_len % self.n_frames_left != 0:
            max_target_len_left = max_target_len + (
                self.n_frames_left - max_target_len % self.n_frames_left
            )
            assert max_target_len_left % self.n_frames_left == 0
        else:
            max_target_len_left = max_target_len

        if max_target_len % self.n_frames_right != 0:
            max_target_len_right = max_target_len + (
                self.n_frames_left - max_target_len % self.n_frames_left
            )
            assert max_target_len_right % self.n_frames_left == 0
        else:
            max_target_len_right = max_target_len

        # include mel padded and gate padded
        mel_padded_left = torch.FloatTensor(
            len(batch), num_mels, max_target_len_left
        )
        mel_padded_left.zero_()
        gate_padded_left = torch.FloatTensor(len(batch), max_target_len_left)
        gate_padded_left.zero_()

        mel_padded_right = torch.FloatTensor(
            len(batch), num_mels, max_target_len_right
        )
        mel_padded_right.zero_()
        gate_padded_right = torch.FloatTensor(len(batch), max_target_len_right)
        gate_padded_right.zero_()

        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_reverse = self.reverse_mel(mel)
            assert mel.size() == mel_reverse.size()
            mel_padded_left[i, :, : mel_reverse.size(1)] = mel_reverse
            gate_padded_left[i, mel_reverse.size(1) :] = 1
            mel_padded_right[i, :, : mel.size(1)] = mel
            gate_padded_right[i, mel.size(1) :] = 1
            output_lengths[i] = mel.size(1)

        mel_padded = {"left": mel_padded_left, "right": mel_padded_right}
        gate_padded = {"left": gate_padded_left, "right": gate_padded_right}

        return (
            text_padded,
            input_lengths,
            mel_padded,
            gate_padded,
            output_lengths,
        )

    def reverse_mel(self, mel):
        mel_dim, mel_seq = mel.size()
        mel_reverse = mel.new_zeros(mel_dim, mel_seq)
        for i in range(mel_seq):
            mel_reverse[:, i] = mel[:, mel_seq - 1 - i]
        return mel_reverse
