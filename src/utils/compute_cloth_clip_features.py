import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().parents[2].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import torch.utils.checkpoint
import torch.utils.checkpoint
import torchvision
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers.utils import check_min_version
from transformers import CLIPVisionModelWithProjection, AutoProcessor, CLIPProcessor
import pickle

from src.dataset.dresscode import DressCodeDataset
from src.dataset.vitonhd import VitonHDDataset
from tqdm import tqdm

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
        default="stabilityai/stable-diffusion-2-inpainting",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument("--test_batch_size", type=int, default=16,
                        help="Batch size (per device) for the testing dataloader.")

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

    parser.add_argument('--dresscode_dataroot', type=str, help='DressCode dataroot')
    parser.add_argument('--vitonhd_dataroot', type=str, help='VitonHD dataroot')

    parser.add_argument("--num_workers_test", type=int, default=8,
                        help="The name of the repository to keep in sync with the local `output_dir`.")

    parser.add_argument("--dataset", type=str, required=True, choices=["dresscode", "vitonhd"], help="dataset to use")
    args = parser.parse_args()

    return args


@torch.no_grad()
def main():
    args = parse_args()

    # Check if the dataset dataroot is provided
    if args.dataset == "vitonhd" and args.vitonhd_dataroot is None:
        raise ValueError("VitonHD dataroot must be provided")
    if args.dataset == "dresscode" and args.dresscode_dataroot is None:
        raise ValueError("DressCode dataroot must be provided")
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device

    # Get the vision encoder and the processor
    if args.pretrained_model_name_or_path == "runwayml/stable-diffusion-inpainting":
        vision_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
        processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    elif args.pretrained_model_name_or_path == "stabilityai/stable-diffusion-2-inpainting":
        vision_encoder = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    else:
        raise ValueError(f"Unknown pretrained model name or path: {args.pretrained_model_name_or_path}")
    vision_encoder.requires_grad_(False)

    vision_encoder = vision_encoder.to(device)
    outputlist = ['cloth', 'c_name']

    # Get the dataset
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
            order='paired',
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
            order='paired',
            radius=5,
            size=(512, 384),
            outputlist=tuple(outputlist)
        )
    else:
        raise NotImplementedError(f"Unknown dataset: {args.dataset}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers_test,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers_test,
    )

    # Extract the CLIP features for the clothes in the dataset and save them to disk.
    save_cloth_features(args.dataset, processor, train_loader, vision_encoder, "train")
    save_cloth_features(args.dataset, processor, test_loader, vision_encoder, "test")


def save_cloth_features(dataset: str, processor: CLIPProcessor, loader: torch.utils.data.DataLoader,
                        vision_encoder: CLIPVisionModelWithProjection, split: str):
    """
    Extract the CLIP features for the clothes in the dataset and save them to disk.
    """
    last_hidden_state_list = []
    cloth_names = []
    for batch in tqdm(loader):
        names = batch["c_name"]
        with torch.cuda.amp.autocast():
            input_image = torchvision.transforms.functional.resize((batch["cloth"] + 1) / 2, (224, 224),
                                                                   antialias=True).clamp(0, 1)
            processed_images = processor(images=input_image, return_tensors="pt")
            visual_features = vision_encoder(processed_images.pixel_values.to(vision_encoder.device))

            last_hidden_state_list.append(visual_features.last_hidden_state.cpu().half())
            cloth_names.extend(names)

    save_dir = PROJECT_ROOT / 'data' / 'clip_cloth_embeddings' / dataset
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.cat(last_hidden_state_list, dim=0), save_dir / f"{split}_last_hidden_state_features.pt")

    with open(os.path.join(save_dir / f"{split}_features_names.pkl"), "wb") as f:
        pickle.dump(cloth_names, f)


if __name__ == '__main__':
    main()
