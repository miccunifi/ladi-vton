import argparse
import json
import os

import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import UNet2DConditionModel, DDIMScheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, AutoProcessor

from dataset.dresscode import DressCodeDataset
from dataset.vitonhd import VitonHDDataset
from models.AutoencoderKL import AutoencoderKL
from models.emasc import EMASC
from models.inversion_adapter import InversionAdapter
from utils.image_from_pipe import generate_images_from_tryon_pipe
from utils.set_seeds import set_seed
from utils.val_metrics import compute_metrics
from vto_pipelines.tryon_pipe import StableDiffusionTryOnePipeline

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB_START_METHOD"] = "thread"


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--dataset", type=str, required=True, choices=["dresscode", "vitonhd"], help="dataset to use")
    parser.add_argument('--dresscode_dataroot', type=str, help='DressCode dataroot')
    parser.add_argument('--vitonhd_dataroot', type=str, help='VitonHD dataroot')
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory",
    )

    parser.add_argument("--save_name", type=str, required=True, help="Name of the saving folder inside output_dir")
    parser.add_argument("--test_order", type=str, required=True, choices=["unpaired", "paired"])


    parser.add_argument("--unet_dir", required=True, type=str, help="Directory where to load the trained unet from")
    parser.add_argument("--unet_name", type=str, default="latest",
                        help="Name of the unet to load from the directory specified by `--unet_dir`. "
                             "To load the latest checkpoint, use `latest`.")

    parser.add_argument(
        "--inversion_adapter_dir", type=str, default=None,
        help="Directory where to load the trained inversion adapter from. Required when using --text_usage=inversion_adapter",
    )
    parser.add_argument("--inversion_adapter_name", type=str, default="latest",
                        help="Name of the inversion adapter to load from the directory specified by `--inversion_adapter_dir`. "
                             "To load the latest checkpoint, use `latest`.")

    parser.add_argument("--emasc_dir", type=str, default=None,
                        help="Directory where to load the trained EMASC from. Required when --emasc_type!=none")
    parser.add_argument("--emasc_name", type=str, default="latest",
                        help="Name of the EMASC to load from the directory specified by `--emasc_dir`. "
                             "To load the latest checkpoint, use `latest`.")


    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-inpainting",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )


    parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")

    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )

    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )

    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )


    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for the dataloader")



    parser.add_argument("--category", type=str, choices=['all', 'lower_body', 'upper_body', 'dresses'], default='all')

    parser.add_argument("--emasc_type", type=str, default='nonlinear', choices=["none", "linear", "nonlinear"],
                        help="Whether to use linear or nonlinear EMASC.")
    parser.add_argument("--emasc_kernel", type=int, default=3, help="EMASC kernel size.")
    parser.add_argument("--emasc_padding", type=int, default=1, help="EMASC padding size.")


    parser.add_argument("--text_usage", type=str, default='inversion_adapter',
                        choices=["none", "noun_chunks", "inversion_adapter"],
                        help="if 'none' do not use the text, if 'noun_chunks' use the coarse noun chunks, if "
                             "'inversion_adapter' use the features obtained trough the inversion adapter net")
    parser.add_argument("--cloth_input_type", type=str, choices=["warped", "none"], default='warped',
                        help="cloth input type. If 'warped' use the warped cloth, if none do not use the cloth as input of the unet")
    parser.add_argument("--num_vstar", default=16, type=int, help="Number of predicted v* images to use")
    parser.add_argument("--num_encoder_layers", default=1, type=int,
                        help="Number of ViT layer to use in inversion adapter")

    parser.add_argument("--use_png", default=False, action="store_true", help="Use png instead of jpg")
    parser.add_argument("--num_inference_steps", default=50, type=int, help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", default=7.5, type=float, help="Guidance scale for the diffusion")
    parser.add_argument("--use_clip_cloth_features", action="store_true",
                        help="Whether to use precomputed clip cloth features")
    parser.add_argument("--compute_metrics", default=False, action="store_true",
                        help="Compute metrics after generation")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


@torch.inference_mode()
def main():
    args = parse_args()

    # Check if the dataset dataroot is provided
    if args.dataset == "vitonhd" and args.vitonhd_dataroot is None:
        raise ValueError("VitonHD dataroot must be provided")
    if args.dataset == "dresscode" and args.dresscode_dataroot is None:
        raise ValueError("DressCode dataroot must be provided")

    # Enable TF32 for faster inference on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Setup accelerator and device.
    accelerator = Accelerator()
    device = accelerator.device

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load scheduler, tokenizer and models.
    val_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    val_scheduler.set_timesteps(args.num_inference_steps, device=device)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Define the extended unet
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

    # Load the trained unet
    if args.unet_name != "latest":
        path = args.unet_name
    else:
        # Get the most recent checkpoint
        dirs = os.listdir(args.unet_dir)
        dirs = [d for d in dirs if d.startswith("unet")]
        dirs = sorted(dirs, key=lambda x: int(os.path.splitext(x.split("_")[-1])[0]))
        path = dirs[-1]
    accelerator.print(f"Loading Unet checkpoint {path}")
    unet.load_state_dict(torch.load(os.path.join(args.unet_dir, path)))

    if args.emasc_type != 'none':
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

        if args.emasc_dir is not None:
            if args.emasc_name != "latest":
                path = args.emasc_name
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(args.emasc_dir)
                dirs = [d for d in dirs if d.startswith("emasc")]
                dirs = sorted(dirs, key=lambda x: int(os.path.splitext(x.split("_")[-1])[0]))
                path = dirs[-1]
            accelerator.print(f"Loading EMASC checkpoint {path}")
            emasc.load_state_dict(torch.load(os.path.join(args.emasc_dir, path)))
        else:
            raise ValueError("No EMASC checkpoint found. Make sure to specify --emasc_dir")
    else:
        emasc = None
        int_layers = None

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # add posemap input to unet
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

        if args.inversion_adapter_dir is not None:
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
            raise ValueError("No inversion adapter checkpoint found. Make sure to specify --inversion_adapter_dir")

        inversion_adapter.requires_grad_(False)

        if args.use_clip_cloth_features:
            outputlist.append('clip_cloth_features')
            vision_encoder = None
        else:
            outputlist.append('cloth')
    else:
        inversion_adapter = None
        vision_encoder = None
        processor = None

    if args.category != 'all':
        category = [args.category]
    else:
        category = ['dresses', 'upper_body', 'lower_body']

    # Define dataset and dataloader
    if args.dataset == "dresscode":
        test_dataset = DressCodeDataset(
            dataroot_path=args.dresscode_dataroot,
            phase='test',
            order=args.test_order,
            radius=5,
            outputlist=outputlist,
            category=category,
            size=(512, 384)
        )
    elif args.dataset == "vitonhd":
        test_dataset = VitonHDDataset(
            dataroot_path=args.vitonhd_dataroot,
            phase='test',
            order=args.test_order,
            radius=5,
            outputlist=outputlist,
            size=(512, 384),
        )
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_dataloader = accelerator.prepare(test_dataloader)

    weight_dtype = torch.float16

    # Move models to device and eval mode
    text_encoder.to(device, dtype=weight_dtype)
    text_encoder.eval()
    vae.to(device, dtype=weight_dtype)
    vae.eval()
    unet.to(device, dtype=weight_dtype)
    unet.eval()
    if emasc is not None:
        emasc.to(device, dtype=weight_dtype)
        emasc.eval()
    if inversion_adapter is not None:
        inversion_adapter.to(device, dtype=weight_dtype)
        inversion_adapter.eval()
    if vision_encoder is not None:
        vision_encoder.to(device, dtype=weight_dtype)
        vision_encoder.eval()

    # Define the pipeline
    val_pipe = StableDiffusionTryOnePipeline(
        text_encoder=text_encoder,
        vae=vae,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=val_scheduler,
        emasc=emasc,
        emasc_int_layers=int_layers,
    ).to(device)

    # Generate images
    with torch.cuda.amp.autocast():
        generate_images_from_tryon_pipe(val_pipe, inversion_adapter, test_dataloader, args.output_dir,
                                        args.test_order, args.save_name, args.text_usage, vision_encoder, processor,
                                        args.cloth_input_type, 1, args.num_vstar, args.seed,
                                        args.num_inference_steps, args.guidance_scale, args.use_png)

    # Compute metrics
    if args.compute_metrics:
        metrics = compute_metrics(
            os.path.join(args.output_dir, f"{args.save_name}_{args.test_order}"), args.test_order,
            args.dataset, args.category, ['all'], args.dresscode_dataroot, args.vitonhd_dataroot)

        with open(os.path.join(args.output_dir, f"metrics_{args.save_name}_{args.test_order}_{args.category}.json"),
                  "w+") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
