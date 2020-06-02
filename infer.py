import os
import argparse
import matplotlib.pylab as plt
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import hparams
from train import load_model
from data_utils import TextMelLoader, TextMelCollate


def plot_data(data, index, path, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(
            data[i], aspect="auto", origin="bottom", interpolation="none"
        )
    file = os.path.join(path, str(index) + ".png")
    plt.savefig(file)


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
    model = load_model(hparams)
    checkpoint_restore = torch.load(args.checkpoint_path)["state_dict"]
    model.load_state_dict(checkpoint_restore)
    model.eval()

    # infer
    batch_parser = model.parse_batch
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader)):
            x, y = batch_parser(batch)
            sequence = x[0]
            (
                mel_outputs,
                mel_outputs_postnet,
                _,
                alignments,
                _,
            ) = model.inference(sequence)

            mel_path = os.path.join(args.output_infer, str(i) + ".pt")
            torch.save(mel_outputs_postnet[0], mel_path)

            plot_data(
                (
                    mel_outputs.data.cpu().numpy()[0],
                    mel_outputs_postnet.data.cpu().numpy()[0],
                    alignments.data.cpu().numpy()[0].T,
                ),
                i,
                args.output_infer,
            )


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
