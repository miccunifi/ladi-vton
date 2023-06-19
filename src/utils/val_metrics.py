import argparse
import json
import os
from typing import List, Tuple, Dict

import PIL.Image
import torch
from cleanfid import fid
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
from tqdm import tqdm

from .generate_fid_stats import make_custom_stats


class GTTestDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot: str, dataset: str, category: str, transform: transforms.Compose):
        """
        Dataset for the ground truth test images
        """

        # Validate inputs
        assert dataset in ['dresscode', 'vitonhd'], 'Unsupported dataset'
        assert category in ['all', 'dresses', 'lower_body', 'upper_body'], 'Unsupported category'

        self.dataset = dataset
        self.category = category
        self.transform = transform
        self.dataroot = dataroot

        # Get the paths to the images
        if dataset == 'dresscode':
            filepath = os.path.join(dataroot, f"test_pairs_paired.txt")
            with open(filepath, 'r') as f:
                lines = f.read().splitlines()

            if category in ['lower_body', 'upper_body', 'dresses']:
                self.paths = sorted(
                    [os.path.join(dataroot, category, 'images', line.strip().split()[0]) for line in lines if
                     os.path.exists(os.path.join(dataroot, category, 'images', line.strip().split()[0]))])
            else:
                self.paths = sorted(
                    [os.path.join(dataroot, category, 'images', line.strip().split()[0]) for line in lines for
                     category in ['lower_body', 'upper_body', 'dresses'] if
                     os.path.exists(os.path.join(dataroot, category, 'images', line.strip().split()[0]))])
        else:  # vitonhd
            filepath = os.path.join(dataroot, f"test_pairs.txt")
            with open(filepath, 'r') as f:
                lines = f.read().splitlines()
            self.paths = sorted([os.path.join(dataroot, 'test', 'image', line.strip().split()[0]) for line in lines])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        name = os.path.splitext(os.path.basename(path))[0]
        img = self.transform(PIL.Image.open(path).convert('RGB'))
        return img, name


class GenTestDataset(torch.utils.data.Dataset):
    def __init__(self, gen_folder: str, category: str, transform: transforms.Compose):
        """
        Dataset for the ground truth test images
        """

        # Validate inputs
        assert category in ['all', 'dresses', 'lower_body', 'upper_body'], 'Unsupported category'

        self.category = category
        self.transform = transform
        self.gen_folder = gen_folder

        # Get the paths to the images
        if category in ['lower_body', 'upper_body', 'dresses']:
            self.paths = sorted(
                [os.path.join(gen_folder, category, name) for name in os.listdir(os.path.join(gen_folder, category))])
        elif category == 'all':
            existing_categories = []
            for category in ['lower_body', 'upper_body', 'dresses']:
                if os.path.exists(os.path.join(gen_folder, category)):
                    existing_categories.append(category)

            self.paths = sorted(
                [os.path.join(gen_folder, category, name) for category in existing_categories for
                 name in os.listdir(os.path.join(gen_folder, category)) if
                 os.path.exists(os.path.join(gen_folder, category, name))])
        else:
            raise ValueError('Unsupported category')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        name = os.path.splitext(os.path.basename(path))[0]
        img = self.transform(PIL.Image.open(path).convert('RGB'))
        return img, name


def compute_metrics(gen_folder: str, test_order: str, dataset: str, category: str, metrics2compute: List[str],
                    dresscode_dataroot: str, vitonhd_dataroot: str, generated_size: Tuple[int, int] = (512, 384),
                    batch_size: int = 32, workers: int = 8) -> Dict[str, float]:
    """
    Computes the metrics for the generated images in gen_folder
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Input validation
    assert test_order in ['paired', 'unpaired']
    assert dataset in ['dresscode', 'vitonhd'], 'Unsupported dataset'
    assert category in ['all', 'dresses', 'lower_body', 'upper_body'], 'Unsupported category'

    if dataset == 'dresscode':
        gt_folder = dresscode_dataroot
    elif dataset == 'vitonhd':
        gt_folder = vitonhd_dataroot
    else:
        raise ValueError('Unsupported dataset')

    for m in metrics2compute:
        assert m in ['all', 'ssim_score', 'lpips_score', 'fid_score', 'kid_score', 'is_score'], 'Unsupported metric'

    if metrics2compute == ['all']:
        metrics2compute = ['ssim_score', 'lpips_score', 'fid_score', 'kid_score', 'is_score']

    # Compute FID and KID scores
    if category == 'all':
        if "fid_score" in metrics2compute or "all" in metrics2compute:
            # Check if FID stats exist, if not compute them
            if not fid.test_stats_exists(f"{dataset}_all", mode='clean'):
                make_custom_stats(dresscode_dataroot, vitonhd_dataroot)

            # Compute FID score
            fid_score = fid.compute_fid(gen_folder, dataset_name=f"{dataset}_all", mode='clean', dataset_split="custom",
                                        verbose=True, use_dataparallel=False)
        if "kid_score" in metrics2compute or "all" in metrics2compute:

            # Check if KID stats exist, if not compute them
            if not fid.test_stats_exists(f"{dataset}_all", mode='clean'):
                make_custom_stats(dresscode_dataroot, vitonhd_dataroot)

            # Compute FID score
            kid_score = fid.compute_kid(gen_folder, dataset_name=f"{dataset}_all", mode='clean', dataset_split="custom",
                                        verbose=True, use_dataparallel=False)
    else:  # single category
        if "fid_score" in metrics2compute or "all" in metrics2compute:

            # Check if FID stats exist, if not compute them
            if not fid.test_stats_exists(f"{dataset}_{category}", mode='clean'):
                make_custom_stats(dresscode_dataroot, vitonhd_dataroot)

            # Compute FID score
            fid_score = fid.compute_fid(os.path.join(gen_folder, category), dataset_name=f"{dataset}_{category}",
                                        mode='clean', verbose=True, dataset_split="custom", use_dataparallel=False)
        if "kid_score" in metrics2compute or "all" in metrics2compute:
            # Check if KID stats exist, if not compute them
            if not fid.test_stats_exists(f"{dataset}_{category}", mode='clean'):
                make_custom_stats(dresscode_dataroot, vitonhd_dataroot)

            # Compute KID score
            kid_score = fid.compute_kid(os.path.join(gen_folder, category),
                                        dataset_name=f"{dataset}_{category}", mode='clean', verbose=True,
                                        dataset_split="custom", use_dataparallel=False)

    # Define transforms, datasets and loaders
    trans = transforms.Compose([
        transforms.Resize(generated_size),
        transforms.ToTensor(),
    ])

    gen_dataset = GenTestDataset(gen_folder, category, transform=trans)
    gt_dataset = GTTestDataset(gt_folder, dataset, category, trans)

    gen_loader = DataLoader(gen_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    gt_loader = DataLoader(gt_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    # Define metrics models
    if "is_score" in metrics2compute or "all" in metrics2compute:
        model_is = InceptionScore(normalize=True).to(device)

    if "ssim_score" in metrics2compute or "all" in metrics2compute:
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    if "lpips_score" in metrics2compute or "all" in metrics2compute:
        lpips = LearnedPerceptualImagePatchSimilarity(net='alex', normalize=True).to(device)

    for idx, (gen_batch, gt_batch) in tqdm(enumerate(zip(gen_loader, gt_loader)), total=len(gt_loader)):
        gen_images, gen_names = gen_batch
        gt_images, gt_names = gt_batch

        assert gen_names == gt_names  # Be sure that the images are in the same order

        gen_images = gen_images.to(device)
        gt_images = gt_images.to(device)

        if "is_score" in metrics2compute or "all" in metrics2compute:
            model_is.update(gen_images)

        if "ssim_score" in metrics2compute or "all" in metrics2compute:
            ssim.update(gen_images, gt_images)

        if "lpips_score" in metrics2compute or "all" in metrics2compute:
            lpips.update(gen_images, gt_images)

    if "is_score" in metrics2compute or "all" in metrics2compute:
        is_score, is_std = model_is.compute()
    if "ssim_score" in metrics2compute or "all" in metrics2compute:
        ssim_score = ssim.compute()
    if "lpips_score" in metrics2compute or "all" in metrics2compute:
        lpips_score = lpips.compute()

    results = {}

    for m in metrics2compute:
        if torch.is_tensor(locals()[m]):
            results[m] = locals()[m].item()
        else:
            results[m] = locals()[m]
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute the metrics for the generated images")
    parser.add_argument("--gen_folder", type=str, required=True, help="Path to the generated images")
    parser.add_argument('--dresscode_dataroot', type=str, help='DressCode dataroot')
    parser.add_argument('--vitonhd_dataroot', type=str, help='VitonHD dataroot')
    parser.add_argument("--test_order", type=str, required=True, choices=['paired', 'unpaired'])
    parser.add_argument("--dataset", type=str, required=True, choices=['dresscode', 'vitonhd'],
                        help="Dataset to use for the metrics")
    parser.add_argument("--category", type=str, choices=['all', 'lower_body', 'upper_body', 'dresses'], default='all')
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the dataloaders")
    parser.add_argument("--workers", type=int, default=8, help="Number of workers for the dataloaders")

    args = parser.parse_args()

    # Check if the dataset dataroot is provided
    if args.dataset == "vitonhd" and args.vitonhd_dataroot is None:
        raise ValueError("VitonHD dataroot must be provided")
    if args.dataset == "dresscode" and args.dresscode_dataroot is None:
        raise ValueError("DressCode dataroot must be provided")

    # Check if the generated images folder exists
    if not os.path.exists(args.gen_folder):
        raise ValueError("The generated images folder does not exist")

    metrics = compute_metrics(args.gen_folder, args.test_order, args.dataset, args.category, ['all'],
                              args.dresscode_dataroot, args.vitonhd_dataroot, batch_size=args.batch_size,
                              workers=args.workers)

    # Print the metrics
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    with open(os.path.join(args.gen_folder, f"metrics_{args.test_order}_{args.category}.json"), "w+") as f:
        json.dump(metrics, f, indent=4)
