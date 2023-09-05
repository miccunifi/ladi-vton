# File based on https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py

import argparse
import itertools
import logging
import os
import shutil

import diffusers
import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.utils.checkpoint
import torchvision
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import DDPMScheduler, UNet2DConditionModel, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, AutoProcessor

from dataset.dresscode import DressCodeDataset
from dataset.vitonhd import VitonHDDataset
from models.AutoencoderKL import AutoencoderKL
from models.inversion_adapter import InversionAdapter
from utils.encode_text_word_embedding import encode_text_word_embedding
from utils.image_from_pipe import generate_images_from_tryon_pipe
from utils.set_seeds import set_seed
from utils.val_metrics import compute_metrics
from vto_pipelines.tryon_pipe import StableDiffusionTryOnePipeline

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB_START_METHOD"] = "thread"

torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parser = argparse.ArgumentParser(description="VTO training script.")
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
        "--inversion_adapter_dir",
        type=str,
        default=None,
        help="Directory where to load the trained inversion adapter from. If not specified, inversion adapter will be trained from scratch.",
    )
    parser.add_argument("--inversion_adapter_name", type=str, default="latest",
                        help="Name of the inversion adapter to load from the directory specified by `--inversion_adapter_dir`. To load the latest checkpoint, use `latest`.")


    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-inpainting",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )


    parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")

    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )

    parser.add_argument(
        "--test_batch_size", type=int, default=16, help="Batch size (per device) for the testing dataloader."
    )

    parser.add_argument("--num_train_epochs", type=int, default=100, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=200001,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
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
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=50000,
        help=(
            "Perform validation step and save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
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
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers to use in the dataloaders.")
    parser.add_argument("--num_workers_test", type=int, default=8,
                        help="Number of workers to use in the test dataloaders.")

    parser.add_argument("--test_order", type=str, default="unpaired", choices=["unpaired", "paired"])
    parser.add_argument("--uncond_fraction", type=float, default=0.2, help="Fraction of unconditioned training samples")
    parser.add_argument("--text_usage", type=str, default='inversion_adapter',
                        choices=["none", "noun_chunks", "inversion_adapter"],
                        help="if 'none' do not use the text, if 'noun_chunks' use the coarse noun chunks, if "
                             "'inversion_adapter' use the features obtained trough the inversion adapter network")
    parser.add_argument("--cloth_input_type", type=str, choices=["warped", "none"], default='warped',
                        help="cloth input type. If 'warped' use the warped cloth, if none do not use the cloth as input of the unet")
    parser.add_argument("--num_vstar", default=16, type=int, help="Number of predicted v* images to use")
    parser.add_argument("--num_encoder_layers", default=1, type=int,
                        help="Number of ViT layer to use in inversion adapter")
    parser.add_argument("--train_inversion_adapter", action="store_true",
                        help="Train also the inversion adapter during training")
    parser.add_argument("--use_clip_cloth_features", action="store_true",
                        help="Whether to use precomputed clip cloth features")

    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()
    inversion_adapter = None

    # Check if the dataset dataroot is provided
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

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    val_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    val_scheduler.set_timesteps(50, device=accelerator.device)

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    new_in_channels = 27 if args.cloth_input_type == "none" else 31
    # the posemap has 18 channels, the (encoded) cloth has 4 channels, the standard SD inpaining has 9 channels
    with torch.no_grad():
        # Replace the first conv layer of the unet with a new one with the correct number of input channels
        conv_new = torch.nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=unet.conv_in.out_channels,
            kernel_size=3,
            padding=1,
        )

        torch.nn.init.kaiming_normal_(conv_new.weight)  # Initialize new conv layer
        conv_new.weight.data = conv_new.weight.data * 0.  # Zero-initialize new conv layer

        conv_new.weight.data[:, :9] = unet.conv_in.weight.data  # Copy weights from old conv layer
        conv_new.bias.data = unet.conv_in.bias.data  # Copy bias from old conv layer

        unet.conv_in = conv_new  # replace conv layer in unet
        unet.config['in_channels'] = new_in_channels  # update config

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    if not args.train_inversion_adapter:
        text_encoder.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        # Enable gradient checkpointing for memory efficient training
        unet.enable_gradient_checkpointing()
        if args.text_usage == "inversion_adapter":
            text_encoder.gradient_checkpointing_enable()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    outputlist = ['image', 'pose_map', 'captions', 'inpaint_mask', 'im_mask', 'category', 'im_name']

    if args.cloth_input_type == 'warped':
        outputlist.append('warped_cloth')

    if args.text_usage == 'inversion_adapter':
        if args.pretrained_model_name_or_path == "runwayml/stable-diffusion-inpainting":
            vision_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
            processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        elif args.pretrained_model_name_or_path == "stabilityai/stable-diffusion-2-inpainting":
            vision_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
            processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        else:
            raise ValueError(f"Unknown pretrained model name or path: {args.pretrained_model_name_or_path}")
        vision_encoder.requires_grad_(False)

        inversion_adapter = InversionAdapter(input_dim=vision_encoder.config.hidden_size,
                                             hidden_dim=vision_encoder.config.hidden_size * 4,
                                             output_dim=text_encoder.config.hidden_size * args.num_vstar,
                                             num_encoder_layers=args.num_encoder_layers,
                                             config=vision_encoder.config)

        if args.inversion_adapter_dir is not None:  # Load inversion adapter checkpoint
            if args.inversion_adapter_name != "latest":
                path = args.inversion_adapter_name
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(args.inversion_adapter_dir)
                dirs = [d for d in dirs if d.startswith("inversion_adapter")]
                dirs = sorted(dirs, key=lambda x: int(os.path.splitext(x.split("_")[-1])[0]))
                path = dirs[-1]
            accelerator.print(f"Loading inversion adapter checkpoint {path}")
            inversion_adapter.load_state_dict(torch.load(os.path.join(args.inversion_adapter_dir, path)))
        else:
            print("No inversion adapter checkpoint found. Training from scratch.", flush=True)

        if args.train_inversion_adapter:
            optimizer.add_param_group({'params': inversion_adapter.parameters(), 'lr': args.learning_rate})
        else:
            inversion_adapter.requires_grad_(False)
            inversion_adapter.eval()

        if args.use_clip_cloth_features:
            outputlist.append('clip_cloth_features')
            vision_encoder = None
        else:
            outputlist.append('cloth')

    # Define datasets and dataloaders.
    if args.dataset == "dresscode":
        train_dataset = DressCodeDataset(
            dataroot_path=args.dresscode_dataroot,
            phase='train',
            order='paired',
            radius=5,
            category=['dresses', 'upper_body', 'lower_body'],
            size=(512, 384),
            outputlist=tuple(outputlist)
        )

        test_dataset = DressCodeDataset(
            dataroot_path=args.dresscode_dataroot,
            phase='test',
            order=args.test_order,
            radius=5,
            category=['dresses', 'upper_body', 'lower_body'],
            size=(512, 384),
            outputlist=tuple(outputlist)
        )
    elif args.dataset == "vitonhd":
        train_dataset = VitonHDDataset(
            dataroot_path=args.vitonhd_dataroot,
            phase='train',
            order='paired',
            radius=5,
            size=(512, 384),
            outputlist=tuple(outputlist)
        )

        test_dataset = VitonHDDataset(
            dataroot_path=args.vitonhd_dataroot,
            phase='test',
            order=args.test_order,
            radius=5,
            size=(512, 384),
            outputlist=tuple(outputlist)
        )
    else:
        raise NotImplementedError(f"Unknown dataset {args.dataset}")

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

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Prepare everything with our `accelerator`.
    if args.text_usage == "inversion_adapter":
        unet, inversion_adapter, text_encoder, optimizer, train_dataloader, lr_scheduler, test_dataloader = accelerator.prepare(
            unet, inversion_adapter, text_encoder, optimizer, train_dataloader, lr_scheduler, test_dataloader
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler, test_dataloader = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler, test_dataloader
        )
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        vision_encoder = None
        processor = None

    # Move and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if args.text_usage == "inversion_adapter" and not args.use_clip_cloth_features:
        vision_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("LaDI_VTON_vto", config=vars(args),
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
        unet.train()
        if inversion_adapter is not None and args.train_inversion_adapter:
            inversion_adapter.train()

        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet), accelerator.accumulate(inversion_adapter):
                # Convert images to latent space
                latents = vae.encode(batch["image"].to(weight_dtype))[0].latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Get the pose map and resize it to the same size as the latents
                pose_map = batch["pose_map"].to(accelerator.device)
                pose_map = torch.nn.functional.interpolate(
                    pose_map, size=(pose_map.shape[2] // 8, pose_map.shape[3] // 8), mode="bilinear")

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                if args.text_usage == "noun_chunks":
                    text = batch["captions"]
                elif args.text_usage == "none":
                    text = [""] * len(batch["captions"])
                elif args.text_usage == 'inversion_adapter':
                    # Generate the text for training the inversion adapter, '$' will be replaced with the PTEs during the
                    # textual encoding process
                    category_text = {
                        'dresses': 'a dress',
                        'upper_body': 'an upper body garment',
                        'lower_body': 'a lower body garment',
                    }
                    text = [f'a photo of a model wearing {category_text[category]} {" $ " * args.num_vstar}' for
                            category in batch['category']]

                    if args.use_clip_cloth_features:
                        # Get the precomputed visual CLIP features of the in-shop cloths
                        clip_cloth_features = batch['clip_cloth_features']
                    else:
                        # Compute the visual features of the in-shop cloths
                        with torch.no_grad():
                            input_image = torchvision.transforms.functional.resize((batch["cloth"] + 1) / 2, (224, 224),
                                                                                   antialias=True).clamp(0, 1)
                            processed_images = processor(images=input_image, return_tensors="pt")
                            clip_cloth_features = vision_encoder(
                                processed_images.pixel_values.to(accelerator.device).to(weight_dtype)).last_hidden_state

                    # Compute the predicted PTEs
                    word_embeddings = inversion_adapter(clip_cloth_features.to(accelerator.device))
                    word_embeddings = word_embeddings.reshape((bsz, args.num_vstar, -1))

                else:
                    raise ValueError(f"Unknown text usage {args.text_usage}")

                target = noise

                # Compute the mask
                mask = batch["inpaint_mask"].to(weight_dtype)
                mask = torch.nn.functional.interpolate(mask, size=(512 // 8, 384 // 8))

                # Get the masked image
                masked_image = batch["im_mask"].to(weight_dtype)
                masked_image_latents = vae.encode(masked_image)[0].latent_dist.sample() * vae.config.scaling_factor

                # Get the warped cloths latents
                if args.cloth_input_type == 'warped':
                    cloth_latents = vae.encode(batch['warped_cloth'].to(weight_dtype))[0].latent_dist.sample()
                elif args.cloth_input_type == 'none':
                    cloth_latents = None
                else:
                    raise ValueError(f"Unknown cloth input type {args.cloth_input_type}")

                if cloth_latents is not None:
                    cloth_latents = cloth_latents * vae.config.scaling_factor

                # Randomly mask out some of the inputs
                if args.uncond_fraction > 0:
                    uncond_mask_text = torch.rand(bsz, device=latents.device) < args.uncond_fraction
                    uncond_mask_cloth = torch.rand(bsz, device=latents.device) < args.uncond_fraction
                    uncond_mask_pose = torch.rand(bsz, device=latents.device) < args.uncond_fraction
                    text = [t if not uncond_mask_text[i] else "" for i, t in enumerate(text)]
                    pose_map[uncond_mask_pose] = torch.zeros_like(pose_map[0])
                    if cloth_latents is not None:
                        cloth_latents[uncond_mask_cloth] = torch.zeros_like(cloth_latents[0])

                # Encode the text
                tokenized_text = tokenizer(text, max_length=tokenizer.model_max_length, padding="max_length",
                                           truncation=True, return_tensors="pt").input_ids
                tokenized_text = tokenized_text.to(accelerator.device)

                # Encode the text
                if args.text_usage in ["inversion_adapter"]:
                    # Encode the text using the PTEs extracted from the in-shop cloths
                    encoder_hidden_states = encode_text_word_embedding(text_encoder, tokenized_text,
                                                                       word_embeddings,
                                                                       args.num_vstar).last_hidden_state
                elif args.text_usage in ["noun_chunks", "none"]:
                    encoder_hidden_states = text_encoder(tokenized_text).last_hidden_state
                else:
                    raise ValueError(f"Unknown text usage {args.text_usage}")

                # Predict the noise residual and compute loss
                if cloth_latents is not None:
                    unet_input = torch.cat(
                        [noisy_latents, mask, masked_image_latents, pose_map.to(weight_dtype), cloth_latents], dim=1)
                else:
                    unet_input = torch.cat(
                        [noisy_latents, mask, masked_image_latents, pose_map.to(weight_dtype)], dim=1)

                model_pred = unet(unet_input, timesteps, encoder_hidden_states).sample

                # loss in accelerator.autocast according to docs https://huggingface.co/docs/accelerate/v0.15.0/quicktour#mixed-precision-training
                with accelerator.autocast():
                    loss = F.mse_loss(model_pred, target, reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if args.train_inversion_adapter:
                        accelerator.clip_grad_norm_(itertools.chain(unet.parameters(), inversion_adapter.parameters()),
                                                    args.max_grad_norm)

                    else:
                        accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)

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
                    unet.eval()
                    if inversion_adapter is not None:
                        inversion_adapter.eval()
                    if accelerator.is_main_process:
                        os.makedirs(os.path.join(args.output_dir, "checkpoint"), exist_ok=True)
                        accelerator_state_path = os.path.join(args.output_dir, "checkpoint",
                                                              f"checkpoint-{global_step}")
                        accelerator.save_state(accelerator_state_path)

                        # Unwrap the Unet
                        unwrapped_unet = accelerator.unwrap_model(unet, keep_fp32_wrapper=True)

                        with torch.no_grad():
                            val_pipe = StableDiffusionTryOnePipeline(
                                text_encoder=text_encoder,
                                vae=vae,
                                unet=unwrapped_unet,
                                tokenizer=tokenizer,
                                scheduler=val_scheduler,
                            ).to(accelerator.device)

                            # Extract the images
                            with torch.cuda.amp.autocast():
                                generate_images_from_tryon_pipe(val_pipe, inversion_adapter, test_dataloader,
                                                                args.output_dir, args.test_order,
                                                                f"imgs_step_{global_step}",
                                                                args.text_usage, vision_encoder, processor,
                                                                args.cloth_input_type, num_vstar=args.num_vstar)

                            # Compute the metrics
                            metrics = compute_metrics(
                                os.path.join(args.output_dir, f"imgs_step_{global_step}_{args.test_order}"),
                                args.test_order,
                                args.dataset, 'all', ['all'], args.dresscode_dataroot, args.vitonhd_dataroot)

                            print(metrics, flush=True)
                            accelerator.log(metrics, step=global_step)

                            # Delete old checkpoints
                            dirs = os.listdir(os.path.join(args.output_dir, "checkpoint"))
                            dirs = [d for d in dirs if d.startswith("checkpoint")]
                            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                            try:
                                path = dirs[-2]
                                shutil.rmtree(os.path.join(args.output_dir, "checkpoint", path), ignore_errors=True)
                            except:
                                print("No checkpoint to delete")

                            # Save the unet
                            unet_path = os.path.join(args.output_dir, f"unet_{global_step}.pth")
                            accelerator.save(unwrapped_unet.state_dict(), unet_path)

                            if args.train_inversion_adapter:
                                # Save the inversion adapter
                                unwrapped_adapter = accelerator.unwrap_model(inversion_adapter, keep_fp32_wrapper=True)
                                inversion_adapter_path = os.path.join(args.output_dir,
                                                                      f"inversion_adapter_{global_step}.pth")
                                accelerator.save(unwrapped_adapter.state_dict(), inversion_adapter_path)

                                del unwrapped_adapter

                            del unwrapped_unet
                            del val_pipe

                        unet.train()
                        inversion_adapter.train()

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # End of training
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
