import os
import time
import argparse
import math
from numpy import finfo

import torch
import torch.nn as nn
from distributed import apply_gradient_allreduce
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from model import Tacotron2, Discriminator
from data_utils import TextMelLoader, TextMelCollate
from loss_function import Tacotron2LossLeft, Tacotron2LossRight
from logger import Tacotron2Logger
import hparams


def lr_decay(learning_rate, iteration):
    if iteration < 50000:
        lr = learning_rate
    elif iteration < 100000:
        lr = 5e-4
    elif iteration < 150000:
        lr = 3e-4
    elif iteration < 200000:
        lr = 1e-4
    else:
        lr = 5e-5

    # lr = learning_rate * 0.96 ** int(iteration / 1000)
    return lr


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        backend=hparams.dist_backend,
        init_method=hparams.dist_url,
        world_size=n_gpus,
        rank=rank,
        group_name=group_name,
    )

    print("Done initializing distributed")


def prepare_dataloaders(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(
        hparams.n_frames_per_step, hparams.n_frames_per_step
    )

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        trainset,
        num_workers=1,
        shuffle=shuffle,
        sampler=train_sampler,
        batch_size=hparams.batch_size,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,
    )
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        os.makedirs(output_directory, exist_ok=True)
        os.chmod(output_directory, 0o775)
        os.makedirs(log_directory, exist_ok=True)
        os.chmod(log_directory, 0o775)
        logger = Tacotron2Logger(log_directory)
    else:
        logger = None
    return logger


def load_model(hparams):
    model = Tacotron2(hparams).cuda()
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo("float16").min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model_dict = checkpoint_dict["state_dict"]
    if len(ignore_layers) > 0:
        model_dict = {
            k: v for k, v in model_dict.items() if k not in ignore_layers
        }
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(
    checkpoint_path, model, optimizer, disc_context, disc_optimizer
):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint_dict["state_dict"])
    optimizer.load_state_dict(checkpoint_dict["optimizer"])
    learning_rate = checkpoint_dict["learning_rate"]
    iteration = checkpoint_dict["iteration"]

    disc_context.load_state_dict(checkpoint_dict["state_dict_disc"])
    disc_optimizer.load_state_dict(checkpoint_dict["optimizer_disc"])

    print(
        "Loaded checkpoint '{}' from iteration {}".format(
            checkpoint_path, iteration
        )
    )
    return (
        model,
        optimizer,
        learning_rate,
        iteration,
        disc_context,
        disc_optimizer,
    )


def save_checkpoint(
    model,
    optimizer,
    learning_rate,
    iteration,
    filepath,
    disc_context,
    disc_optimizer,
):
    print(
        "Saving model and optimizer state at iteration {} to {}".format(
            iteration, filepath
        )
    )
    torch.save(
        {
            "iteration": iteration,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
            "state_dict_disc": disc_context.state_dict(),
            "optimizer_disc": disc_optimizer.state_dict(),
        },
        filepath,
    )


def validate(
    model,
    disc_context,
    criterion_left,
    criterion_right,
    disc_criterion,
    valset,
    batch_size,
    n_gpus,
    collate_fn,
    distributed_run,
):
    """Handles all the validation scoring and printing"""
    model.eval()
    disc_context.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(
            valset,
            sampler=val_sampler,
            num_workers=1,
            shuffle=False,
            batch_size=batch_size,
            pin_memory=False,
            collate_fn=collate_fn,
        )

        val_loss = 0.0
        val_loss_left = 0.0
        val_loss_right = 0.0
        val_disc_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            [outs_left, outs_right] = model(x, teacher=True)
            y_pred_left = outs_left[0:3]
            y_pred_right = outs_right[0:3]

            # evaluate the discriminator on validation dataset
            att_left = outs_left[-1]
            att_right = outs_right[-1]
            att_num = att_left.size(0)
            disc_label_left = torch.full((att_num, 1), 0.0).cuda()
            disc_label_right = torch.full((att_num, 1), 1.0).cuda()
            disc_out_left = disc_context(att_left.detach())
            disc_out_right = disc_context(att_right.detach())
            disc_loss_left = disc_criterion(disc_out_left, disc_label_left)
            disc_loss_right = disc_criterion(disc_out_right, disc_label_right)
            disc_loss = 0.5 * (disc_loss_left + disc_loss_right)

            loss_left = criterion_left(y_pred_left, y)
            loss_right = criterion_right(y_pred_right, y)
            loss = loss_left + loss_right

            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()

            val_loss += reduced_val_loss
            val_loss_left += loss_left.item()
            val_loss_right += loss_right.item()
            val_disc_loss += disc_loss.item()

        val_loss = val_loss / (i + 1)
        val_loss_left = val_loss_left / (i + 1)
        val_loss_right = val_loss_right / (i + 1)
        val_disc_loss = val_disc_loss / (i + 1)

    model.train()
    disc_context.train()
    return val_loss, val_loss_left, val_loss_right, val_disc_loss


def train(
    output_directory,
    log_directory,
    checkpoint_path,
    warm_start,
    n_gpus,
    rank,
    group_name,
    hparams,
):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_directory (string): directory to save checkpoints
    log_directory (string) directory to save tensorboard logs
    checkpoint_path(string): checkpoint path for restore
    n_gpus (int): number of gpus
    rank (int): rank of current gpu
    hparams (object): comma separated list of "name=value" pairs.
    """

    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=hparams.weight_decay
    )

    if hparams.fp16_run:
        from apex import amp

        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    # 0. add a discriminator model and its optimiser
    disc_context = Discriminator(
        in_size=hparams.disc_in,
        hid_size=hparams.disc_hid,
        out_size=hparams.disc_out,
    ).cuda()
    disc_optimizer = torch.optim.Adam(disc_context.parameters(), lr=0.0001)

    criterion_left = Tacotron2LossLeft()
    criterion_right = Tacotron2LossRight()
    disc_criterion = nn.BCELoss()
    logger = prepare_directories_and_logger(
        output_directory, log_directory, rank
    )

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint if one exists
    iteration = 0
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(
                checkpoint_path, model, hparams.ignore_layers
            )
        else:
            (
                model,
                optimizer,
                _learning_rate,
                iteration,
                disc_context,
                disc_optimizer,
            ) = load_checkpoint(
                checkpoint_path, model, optimizer, disc_context, disc_optimizer
            )
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1  # next iteration is iteration + 1
            epoch_offset = max(0, int(iteration / len(train_loader)))

    model.train()
    disc_context.train()
    is_overflow = False

    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            learning_rate = lr_decay(hparams.learning_rate, iteration)
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

            model.zero_grad()
            disc_context.zero_grad()
            x, y = model.parse_batch(batch)
            y_pred = model(x, teacher=True)
            [outs_left, outs_right] = y_pred
            y_pred_left = outs_left[0:3]
            y_pred_right = outs_right[0:3]

            # 1. load the context seq for both left and right decoder
            att_left = outs_left[-1]
            att_right = outs_right[-1]
            att_num = att_left.size(0)

            # 2. set real label for left decoder and right decoder
            disc_label_left = torch.full((att_num, 1), 0.0).cuda()
            disc_label_right = torch.full((att_num, 1), 1.0).cuda()

            # 3. train the discriminator
            disc_out_left = disc_context(att_left.detach())
            disc_loss_left = disc_criterion(disc_out_left, disc_label_left)
            disc_loss_left.backward()

            disc_out_right = disc_context(att_right.detach())
            disc_loss_right = disc_criterion(disc_out_right, disc_label_right)
            disc_loss_right.backward()

            # 4. update the parameters of discriminator and add log
            disc_loss = 0.5 * (disc_loss_left + disc_loss_right)
            logger.log_assist(
                "loss of discriminator for each iteration",
                float(disc_loss.item()),
                iteration,
            )

            cmp_left = torch.ge(disc_out_left, 0.5).float()
            res_left = torch.sum(torch.eq(cmp_left, disc_label_left)).float()
            acc_left = res_left / float(disc_out_left.numel())
            logger.log_assist(
                "acc of discriminator for left decoder",
                float(acc_left.item()),
                iteration,
            )

            cmp_right = torch.ge(disc_out_right, 0.5).float()
            res_right = torch.sum(
                torch.eq(cmp_right, disc_label_right)
            ).float()
            acc_right = res_right / float(disc_out_right.numel())
            logger.log_assist(
                "acc of discriminator for right decoder",
                float(acc_right.item()),
                iteration,
            )

            acc = (res_left + res_right) / (
                float(disc_out_left.numel()) + float(disc_out_right.numel())
            )
            logger.log_assist(
                "acc of discriminator for each iteration",
                float(acc.item()),
                iteration,
            )
            if acc <= 0.95:
                disc_optimizer.step()

            # 5. train the generator to reconstruct the mels (original part)
            loss_left = criterion_left(y_pred_left, y)
            loss_right = criterion_right(y_pred_right, y)
            loss = loss_left + loss_right

            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()
            if hparams.fp16_run:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # 6. train the generator to fool the discriminator (new adding)
            disc_label_left_g = torch.full((att_num, 1), 1.0).cuda()
            disc_label_right_g = torch.full((att_num, 1), 0.0).cuda()
            disc_out_left_g = disc_context(att_left)
            disc_out_right_g = disc_context(att_right)
            loss_left_g = disc_criterion(disc_out_left_g, disc_label_left_g)
            loss_right_g = disc_criterion(disc_out_right_g, disc_label_right_g)
            loss_g = 0.5 * (loss_left_g + loss_right_g)
            logger.log_assist(
                "loss of genarator for each iteration",
                float(loss_g.item()),
                iteration,
            )

            # if acc >= 0.75:
            #    loss_g.backward()

            # 7. check the gradient of the model and update parameters
            if hparams.fp16_run:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), hparams.grad_clip_thresh
                )
                is_overflow = math.isnan(grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh
                )
            optimizer.step()

            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                print(
                    "Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(
                        iteration, reduced_loss, grad_norm, duration
                    )
                )
                logger.log_training(
                    reduced_loss, grad_norm, learning_rate, duration, iteration
                )
                logger.log_assist(
                    "loss for left forward decoder",
                    loss_left.item(),
                    iteration,
                )
                logger.log_assist(
                    "loss for right forward decoder",
                    loss_right.item(),
                    iteration,
                )

            if not is_overflow and (
                iteration % hparams.iters_per_checkpoint == 0
            ):
                (
                    reduced_val_loss,
                    val_loss_left,
                    val_loss_right,
                    val_disc_loss,
                ) = validate(
                    model,
                    disc_context,
                    criterion_left,
                    criterion_right,
                    disc_criterion,
                    valset,
                    hparams.batch_size,
                    n_gpus,
                    collate_fn,
                    hparams.distributed_run,
                )

                if rank == 0:
                    print(
                        "Validation loss {}: {:9f}  ".format(
                            iteration, reduced_val_loss
                        )
                    )
                    logger.log_validation(
                        reduced_val_loss, model, y, y_pred, iteration,
                    )
                    logger.log_assist(
                        "valid loss for left forward decoder",
                        val_loss_left,
                        iteration,
                    )
                    logger.log_assist(
                        "valid loss for right forward decoder",
                        val_loss_right,
                        iteration,
                    )
                    logger.log_assist(
                        "valid loss for the discriminator",
                        val_disc_loss,
                        iteration,
                    )

                    checkpoint_path = os.path.join(
                        output_directory, "checkpoint_{}".format(iteration)
                    )
                    save_checkpoint(
                        model,
                        optimizer,
                        learning_rate,
                        iteration,
                        checkpoint_path,
                        disc_context,
                        disc_optimizer,
                    )

            iteration += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output_directory",
        type=str,
        default="/opt/checkpoints/tacotron2/exp2",
        help="directory to save checkpoints",
    )
    parser.add_argument(
        "-l",
        "--log_directory",
        type=str,
        default="/opt/checkpoints/tacotron2/log/exp2",
        help="directory to save tensorboard logs",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        default=None,
        required=False,
        help="checkpoint path",
    )
    parser.add_argument(
        "--warm_start",
        action="store_true",
        help="load model weights only, ignore specified layers",
    )
    parser.add_argument(
        "--n_gpus", type=int, default=1, required=False, help="number of gpus"
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        required=False,
        help="rank of current gpu",
    )
    parser.add_argument(
        "--group_name",
        type=str,
        default="group_name",
        required=False,
        help="Distributed group name",
    )

    args = parser.parse_args()

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("FP16 Run:", hparams.fp16_run)
    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    train(
        args.output_directory,
        args.log_directory,
        args.checkpoint_path,
        args.warm_start,
        args.n_gpus,
        args.rank,
        args.group_name,
        hparams,
    )
