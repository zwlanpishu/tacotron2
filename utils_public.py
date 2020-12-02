import torch

import numpy as np
import librosa

# from scipy.io.wavfile import read

import hparams as hparams
from utils_audio import preemphasis


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device)
    mask = ids < lengths.unsqueeze(1)
    return mask


def load_wav_to_torch(full_path, sr):
    data, sampling_rate = librosa.load(full_path, sr=sr)
    data, _ = librosa.effects.trim(data, top_db=55)
    if hparams.tacotron1_norm is True:
        data = preemphasis(data)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x, gpu=0):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(gpu)
    return x


def parse_batch(batch, gpu=0):
    (text_padded, input_lengths, mel_padded, gate_padded, output_lengths,) = batch
    text_padded = to_gpu(text_padded, gpu).long()
    input_lengths = to_gpu(input_lengths, gpu).long()
    max_len = torch.max(input_lengths.data).item()
    mel_padded = to_gpu(mel_padded, gpu).float()
    gate_padded = to_gpu(gate_padded, gpu).float()
    output_lengths = to_gpu(output_lengths, gpu).long()

    return (
        (text_padded, input_lengths, mel_padded, max_len, output_lengths),
        (mel_padded, gate_padded),
    )
