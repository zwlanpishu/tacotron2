import os
import argparse
import matplotlib.pylab as plt
import time
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import hparams
from model import Tacotron2
from utils_data import TextMelLoader, TextMelCollate
from utils_public import parse_batch


def plot_data(data, index, path, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect="auto", origin="bottom", interpolation="none")
    file = os.path.join(path, str(index) + ".png")
    plt.savefig(file)
    plt.close()


def denormalize_feats(feat, cmvn_path):
    feat = feat.detach().cpu().numpy()
    cmvn = np.load(os.path.join(cmvn_path, "cmvn.npy"))
    mean = cmvn[:, 0:1]
    std = cmvn[:, 1:]
    feat = (feat * std) + mean
    feat = torch.from_numpy(feat)
    return feat


def main(args, hparams):

    # prepare data
    testset = TextMelLoader(hparams.test_files, hparams, shuffle=False)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)
    test_loader = DataLoader(
        testset,
        num_workers=1,
        shuffle=False,
        batch_size=1,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    # prepare model
    model = Tacotron2(hparams).cuda(device="cuda:0")
    checkpoint_restore = torch.load(args.checkpoint_path)["state_dict"]
    model.load_state_dict(checkpoint_restore)
    model.eval()
    print("# total parameters:", sum(p.numel() for p in model.parameters()))

    # infer
    duration_add = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader)):
            x, y = parse_batch(batch)

            # the start time
            start = time.perf_counter()
            (mel_outputs, mel_outputs_postnet, _, alignments, _,) = model.inference(x)

            # the end time
            duration = time.perf_counter() - start
            duration_add += duration

            # denormalize the feats and save the mels and attention plots
            mel_predict = mel_outputs_postnet[0]
            mel_denorm = denormalize_feats(mel_predict, hparams.dump)
            mel_path = os.path.join(args.output_infer, "{:0>3d}".format(i) + ".pt")
            torch.save(mel_denorm, mel_path)

            plot_data(
                (
                    mel_outputs.detach().cpu().numpy()[0],
                    mel_outputs_postnet.detach().cpu().numpy()[0],
                    alignments.detach().cpu().numpy()[0].T,
                    mel_denorm.numpy(),
                ),
                i,
                args.output_infer,
            )

        duration_avg = duration_add / (i + 1)
        print("The average inference time is: %f" % duration_avg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_infer",
        type=str,
        default="output_infer",
        help="directory to save infer outputs",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        default=None,
        required=False,
        help="checkpoint path for infer model",
    )
    args = parser.parse_args()
    os.makedirs(args.output_infer, exist_ok=True)
    assert args.checkpoint_path is not None

    main(args, hparams)
    print("finished")
