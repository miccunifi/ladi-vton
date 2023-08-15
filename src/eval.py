import argparse
import json
import os

import accelerate
import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import UNet2DConditionModel, DDIMScheduler
from models.AutoencoderKL import AutoencoderKL
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from dataset.dresscode import DressCodeDataset
from dataset.vitonhd import VitonHDDataset
from models.inversion_adapter import InversionAdapter

from vto_pipelines.tryon_pipe import StableDiffusionTryOnePipeline
from utils.image_from_pipe import generate_images_from_tryon_pipe
from utils.set_seeds import set_seed
from utils.val_metrics import compute_metrics
from models.emasc import EMASC

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB_START_METHOD"] = "thread"


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")

    parser.add_argument(
        "--test_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
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
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    parser.add_argument('--dresscode_dataroot', type=str, help='DressCode dataroot')
    parser.add_argument('--vitonhd_dataroot', type=str, help='VitonHD dataroot')

    parser.add_argument("--num_workers", type=int, default=8,
                        help="The name of the repository to keep in sync with the local `output_dir`.",
                        )

    parser.add_argument("--num_workers_test", type=int, default=8,
                        help="The name of the repository to keep in sync with the local `output_dir`.",
                        )

    parser.add_argument(
        "--inversion_adapter_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--inversion_adapter_name", type=str, default="latest")

    parser.add_argument("--test_order", type=str, required=True, choices=["unpaired", "paired"])
    parser.add_argument("--mask_type", type=str, required=True, choices=["keypoints", "bounding_box"])
    parser.add_argument("--save_name", type=str, required=True)
    parser.add_argument("--cloth_cond_rate", type=float, required=True)
    parser.add_argument("--skip_image_extraction", action="store_true")
    parser.add_argument("--dataset", type=str, required=True, choices=["dresscode", "vitonhd"], help="dataset to use")
    parser.add_argument("--refinement", action="store_true", help="Use if you want to use refined warped garments")
    parser.add_argument("--no_pose", action="store_true")
    parser.add_argument("--category", type=str, choices=['all', 'lower_body', 'upper_body', 'dresses'], default='all')
    parser.add_argument(
        "--skip_type",
        type=str,
        required=True,
        choices=["none", "linear", "nonlinear"],
    )
    parser.add_argument("--skip_kernel", type=int, default=3)
    parser.add_argument("--skip_padding", type=int, default=1)
    parser.add_argument(
        "--skip_input", action="store_true", help="Use skip connection on input."
    )
    parser.add_argument(
        "--skip_conv", action="store_true", help="Use skip after first conv."
    )
    parser.add_argument("--skip_adapter_path", type=str, default='')
    parser.add_argument("--text_usage", type=str,
                        choices=["none", "noun_chunks", "word_embeddings", "inversion_adapter"],
                        help="if 'none' do not use the text, if 'noun_chunks' use the coarse noun chunks, if "
                             "'word_embedding' use the word embedding obtained trough textual inversion, if "
                             "'inversion_adapter' use the features obtained trough the inversion adapter net")
    parser.add_argument("--cloth_input_type", required=True, type=str, choices=["in_shop", "warped", "both", "none"])
    parser.add_argument("--adapter_type", type=str, choices=["mlp", "encoder"], default=None)
    parser.add_argument("--num_vstar", default=1, type=int)
    parser.add_argument("--num_encoder_layers", default=1, type=int)
    parser.add_argument("--use_png", default=False, action="store_true")
    parser.add_argument("--num_inference_steps", default=50, type=int)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()
    inversion_adapter = None
    warped_clothes_type = "warped_cloth_ref" if args.refinement else "warped_cloth_tps"
    if not args.refinement:
        print("STAI USANDO LA TPS NON REFINED !!!!!!!!, CAZZO FAI????????")

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    device = accelerator.device

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load scheduler, tokenize r and models.
    val_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    val_scheduler.set_timesteps(50, device=device)

    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")

    in_feature_channels = [128, 128, 256, 512]
    out_feature_channels = [256, 512, 512, 512]

    int_layers = [2, 3, 4, 5]

    if args.skip_conv:
        int_layers.append(1)
        in_feature_channels.insert(0, 128)
        out_feature_channels.insert(0, 128)
    if args.skip_input:
        int_layers.append(0)
        in_feature_channels.insert(0, 3)
        out_feature_channels.insert(0, 3)

    int_layers.sort()

    if args.skip_type == "none":
        sk_adapter = None
    else:
        sk_adapter = EMASC(in_feature_channels,
                           out_feature_channels,
                           kernel_size=args.skip_kernel,
                           padding=args.skip_padding,
                           stride=1,
                           skip_layers=int_layers,
                           type=args.skip_type)

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
    )

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    new_in_channels = 31 if args.cloth_input_type != "both" else 35
    new_in_channels = 27 if args.cloth_input_type == "none" else new_in_channels
    with torch.no_grad():
        # create new conv layer with unet channels + posemap channels
        conv_new = torch.nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=unet.conv_in.out_channels,
            kernel_size=3,
            padding=1,
        )

        torch.nn.init.kaiming_normal_(conv_new.weight)  # Initialize new conv layer
        conv_new.weight.data = conv_new.weight.data

        conv_new.weight.data[:, :9] = unet.conv_in.weight.data
        conv_new.bias.data = unet.conv_in.bias.data

        unet.conv_in = conv_new  # replace conv layer in unet
        unet.config['in_channels'] = new_in_channels  # update config

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    accelerate.load_checkpoint_in_model(unet, os.path.join(args.output_dir, 'best_pipeline', 'unet',
                                                           'diffusion_pytorch_model.bin'))

    if args.skip_type != 'none':
        sk_adapter.load_state_dict(torch.load(args.skip_adapter_path, map_location='cuda:0'))

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # add posemap input to unet

    outputlist = ['image', 'pose_map', 'original_captions', 'inpaint_mask', 'im_mask', 'category', 'im_name', 'cloth']

    if args.cloth_input_type in ['both', 'warped']:
        outputlist.append(warped_clothes_type)
    if args.cloth_input_type in ['both', 'in_shop']:
        outputlist.append('cloth')

    if args.text_usage == 'word_embeddings':
        outputlist.append('word_embedding')
    elif args.text_usage == 'inversion_adapter':
        outputlist.append('clip_cloth_features')

        if args.pretrained_model_name_or_path == "runwayml/stable-diffusion-inpainting":
            vision_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
        elif args.pretrained_model_name_or_path == "stabilityai/stable-diffusion-2-inpainting":
            vision_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        else:
            raise ValueError(f"Unknown pretrained model name or path: {args.pretrained_model_name_or_path}")
        vision_encoder.requires_grad_(False)

        if args.adapter_type == "mlp":
            inversion_adapter = InversionAdapterMLP(input_dim=text_encoder.config.hidden_size,
                                                    hidden_dim=text_encoder.config.hidden_size * 4,
                                                    output_dim=text_encoder.config.hidden_size * args.num_vstar)
        elif args.adapter_type == "encoder":
            inversion_adapter = InversionAdapter(input_dim=vision_encoder.config.hidden_size,
                                                 hidden_dim=vision_encoder.config.hidden_size * 4,
                                                 output_dim=text_encoder.config.hidden_size * args.num_vstar,
                                                 num_encoder_layers=args.num_encoder_layers,
                                                 config=vision_encoder.config)
        else:
            raise ValueError(f"Unknown adapter type: {args.adapter_type}")
        if args.inversion_adapter_dir is not None:
            if args.inversion_adapter_name != "latest":
                path = args.inversion_adapter_name
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(args.inversion_adapter_dir)
                dirs = [d for d in dirs if d.startswith("inversion_adapter")]
                dirs = sorted(dirs, key=lambda x: int(os.path.splitext(x.split("_")[-1])[0]))
                path = dirs[-1]
            accelerator.print(f"Resuming from checkpoint {path}")
            inversion_adapter.load_state_dict(torch.load(os.path.join(args.inversion_adapter_dir, path)))
        else:
            print("No inversion adapter checkpoint found. Training from scratch.", flush=True)

        inversion_adapter.requires_grad_(False)
        del vision_encoder

    if args.category != 'all':
        category = [args.category]
    else:
        category = ['dresses', 'upper_body', 'lower_body']

    if args.dataset == "dresscode":
        test_dataset = DressCodeDataset(
            args,
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
            args,
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
        batch_size=args.test_batch_size,
        num_workers=args.num_workers_test,
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if args.mixed_precision == 'fp16':
        weight_dtype = torch.float16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    if sk_adapter:
        sk_adapter.to(device, dtype=weight_dtype)

    if inversion_adapter is not None:
        inversion_adapter.to(device, dtype=weight_dtype)
    # Move text_encode and vae to gpu and cast to weight_dtype

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.

    unet.eval()
    with torch.inference_mode():
        val_pipe = StableDiffusionTryOnePipeline(
            text_encoder=text_encoder,
            vae=vae,
            tokenizer=tokenizer,
            unet=unet.to(vae.dtype),
            scheduler=val_scheduler,
            emasc=sk_adapter,
            emasc_int_layers=int_layers,
        ).to(device)

        # val_pipe.enable_attention_slicing()
        test_dataloader = accelerator.prepare(test_dataloader)
        if not args.skip_image_extraction:
            with torch.cuda.amp.autocast():
                generate_images_from_tryon_pipe(args, val_pipe, test_dataloader, inversion_adapter)
        metrics = compute_metrics(
            os.path.join(args.output_dir, f"{args.save_name}_{args.test_order}"), args.test_order,
            args.dataset, args.category, ['all'], args.dresscode_dataroot, args.vitonhd_dataroot)

        with open(os.path.join(args.output_dir, f"{args.save_name}metrics_{args.test_order}_{args.category}.json"),
                  "w+") as f:
            json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
