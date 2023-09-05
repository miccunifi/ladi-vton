# File based on https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py

import argparse
import logging
import os
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()

import diffusers
import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from tqdm.auto import tqdm

from dataset.dresscode import DressCodeDataset
from dataset.vitonhd import VitonHDDataset
from models.AutoencoderKL import AutoencoderKL
from models.emasc import EMASC
from utils.data_utils import mask_features
from utils.image_from_pipe import extract_save_vae_images
from utils.set_seeds import set_seed
from utils.val_metrics import compute_metrics
from utils.vgg_loss import VGGLoss

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB_START_METHOD"] = "thread"


def parse_args():
    parser = argparse.ArgumentParser(description="EMASC training script")
    parser.add_argument("--dataset", type=str, required=True, choices=["dresscode", "vitonhd"], help="dataset to use")
    parser.add_argument('--dresscode_dataroot', type=str, help='DressCode dataroot')
    parser.add_argument('--vitonhd_dataroot', type=str, help='VitonHD dataroot')
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-inpainting",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )


    parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")

    parser.add_argument(
        "--train_batch_size", type=int, default=16, help=" Batch size (per device) for the training dataloader."
    )

    parser.add_argument(
        "--test_batch_size", type=int, default=16, help="Batch size (per device) for the testing dataloader."
    )

    parser.add_argument("--num_train_epochs", type=int, default=100, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=40001,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default='fp16',
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ', `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=10000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader number of workers.")
    parser.add_argument("--num_workers_test", type=int, default=8, help="Test DataLoader number of workers.")
    parser.add_argument("--test_order", type=str, default="paired", choices=["unpaired", "paired"],
                        help="Whether to use paired or unpaired test data.")
    parser.add_argument("--emasc_type", type=str, default='nonlinear', choices=["linear", "nonlinear"],
                        help="Whether to use linear or nonlinear EMASC.")
    parser.add_argument('--vgg_weight', type=float, default=0.5, help='weight for vgg loss')
    parser.add_argument("--emasc_kernel", type=int, default=3, help="EMASC kernel size.")
    parser.add_argument("--emasc_padding", type=int, default=1, help="EMASC padding size.")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()

    if args.dataset == "vitonhd" and args.vitonhd_dataroot is None:
        raise ValueError("VitonHD dataroot must be provided")
    if args.dataset == "dresscode" and args.dresscode_dataroot is None:
        raise ValueError("DressCode dataroot must be provided")

    # Setup accelerator.
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load VAE model.
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae.eval()

    # Define EMASC model.
    in_feature_channels = [128, 128, 128, 256, 512]
    out_feature_channels = [128, 256, 512, 512, 512]
    int_layers = [1, 2, 3, 4, 5]

    emasc = EMASC(in_feature_channels,
                  out_feature_channels,
                  kernel_size=args.emasc_kernel,
                  padding=args.emasc_padding,
                  stride=1,
                  type=args.emasc_type)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        emasc.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Define datasets and dataloaders.
    if args.dataset == "dresscode":
        train_dataset = DressCodeDataset(
            dataroot_path=args.dresscode_dataroot,
            phase='train',
            order='paired',
            radius=5,
            category=['dresses', 'upper_body', 'lower_body'],
            size=(512, 384),
            outputlist=('image', 'pose_map', 'inpaint_mask', 'im_mask', 'category', 'im_name')
        )

        test_dataset = DressCodeDataset(
            dataroot_path=args.dresscode_dataroot,
            phase='test',
            order=args.test_order,
            radius=5,
            category=['dresses', 'upper_body', 'lower_body'],
            size=(512, 384),
            outputlist=('image', 'pose_map', 'inpaint_mask', 'im_mask', 'category', 'im_name')
        )
    elif args.dataset == "vitonhd":
        train_dataset = VitonHDDataset(
            dataroot_path=args.vitonhd_dataroot,
            phase='train',
            order='paired',
            radius=5,
            size=(512, 384),
            outputlist=('image', 'pose_map', 'inpaint_mask', 'im_mask', 'category', 'im_name')
        )

        test_dataset = VitonHDDataset(
            dataroot_path=args.vitonhd_dataroot,
            phase='test',
            order=args.test_order,
            radius=5,
            size=(512, 384),
            outputlist=('image', 'pose_map', 'inpaint_mask', 'im_mask', 'category', 'im_name')
        )
    else:
        raise NotImplementedError("Dataset not implemented")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers_test,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Define VGG loss when vgg_weight > 0
    if args.vgg_weight > 0:
        criterion_vgg = VGGLoss()
    else:
        criterion_vgg = None

    # Prepare everything with our `accelerator`.
    emasc, vae, train_dataloader, lr_scheduler, test_dataloader, criterion_vgg = accelerator.prepare(
        emasc, vae, train_dataloader, lr_scheduler, test_dataloader, criterion_vgg)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("LaDI_VTON_EMASC", config=vars(args),
                                  init_kwargs={"wandb": {"name": os.path.basename(args.output_dir)}})
        if args.report_to == 'wandb':
            wandb_tracker = accelerator.get_tracker("wandb")
            wandb_tracker.name = os.path.basename(args.output_dir)
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        try:
            if args.resume_from_checkpoint != "latest":
                path = os.path.basename(os.path.join("checkpoint", args.resume_from_checkpoint))
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(os.path.join(args.output_dir, "checkpoint"))
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1]
            accelerator.print(f"Resuming from checkpoint {path}")

            accelerator.load_state(os.path.join(args.output_dir, "checkpoint", path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = global_step % num_update_steps_per_epoch
        except Exception as e:
            print("Failed to load checkpoint, training from scratch:")
            print(e)
            resume_step = 0

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        emasc.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(emasc):
                # Convert images to latent space
                with torch.no_grad():
                    # take latents from the encoded image and intermediate features from the encoded masked image
                    posterior_im, _ = vae.encode(batch["image"])
                    _, intermediate_features = vae.encode(batch["im_mask"])

                    intermediate_features = [intermediate_features[i] for i in int_layers]

                # Use EMASC to process the intermediate features
                processed_intermediate_features = emasc(intermediate_features)

                # Mask the features
                processed_intermediate_features = mask_features(processed_intermediate_features, batch["inpaint_mask"])

                # Decode the image from the latent space use the EMASC module
                latents = posterior_im.latent_dist.sample()
                reconstructed_image = vae.decode(z=latents,
                                                 intermediate_features=processed_intermediate_features,
                                                 int_layers=int_layers).sample

                # Compute the loss
                with accelerator.autocast():
                    loss = F.l1_loss(reconstructed_image, batch["image"], reduction="mean")
                    if criterion_vgg:
                        loss += args.vgg_weight * (criterion_vgg(reconstructed_image, batch["image"]))

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate and update gradients
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(emasc.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                # Save checkpoint every checkpointing_steps steps
                if global_step % args.checkpointing_steps == 0:
                    # Validation Step
                    emasc.eval()
                    if accelerator.is_main_process:
                        # Save model checkpoint
                        os.makedirs(os.path.join(args.output_dir, "checkpoint"), exist_ok=True)
                        accelerator_state_path = os.path.join(args.output_dir, "checkpoint",
                                                              f"checkpoint-{global_step}")
                        accelerator.save_state(accelerator_state_path)

                        # Unwrap the EMASC model
                        unwrapped_emasc = accelerator.unwrap_model(emasc, keep_fp32_wrapper=True)
                        with torch.no_grad():
                            # Extract the images
                            with torch.cuda.amp.autocast():
                                extract_save_vae_images(vae, unwrapped_emasc, test_dataloader, int_layers,
                                                        args.output_dir, args.test_order,
                                                        save_name=f"imgs_step_{global_step}",
                                                        emasc_type=args.emasc_type)

                            # Compute the metrics
                            metrics = compute_metrics(
                                os.path.join(args.output_dir, f"imgs_step_{global_step}_{args.test_order}"),
                                args.test_order, args.dataset, 'all', ['all'], args.dresscode_dataroot,
                                args.vitonhd_dataroot)

                            print(metrics, flush=True)
                            accelerator.log(metrics, step=global_step)

                            # Delete the previous checkpoint
                            dirs = os.listdir(os.path.join(args.output_dir, "checkpoint"))
                            dirs = [d for d in dirs if d.startswith("checkpoint")]
                            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                            try:
                                path = dirs[-2]
                                shutil.rmtree(os.path.join(args.output_dir, "checkpoint", path), ignore_errors=True)
                            except:
                                print("No checkpoint to delete")

                            # Save EMASC model
                            emasc_path = os.path.join(args.output_dir, f"emasc_{global_step}.pth")
                            accelerator.save(unwrapped_emasc.state_dict(), emasc_path)
                            del unwrapped_emasc

                        emasc.train()

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # End of training
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
