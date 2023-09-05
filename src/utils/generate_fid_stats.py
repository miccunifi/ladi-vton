import argparse
import os
import shutil

from cleanfid import fid
from tqdm import tqdm


def make_custom_stats(dresscode_dataroot: str, vitonhd_dataroot: str):
    if dresscode_dataroot is not None:
        dresscode_filesplit = os.path.join(dresscode_dataroot, f"test_pairs_paired.txt")
        with open(dresscode_filesplit, 'r') as f:
            lines = f.read().splitlines()
        for category in ['lower_body', 'upper_body', 'dresses']:
            if not fid.test_stats_exists(f"dresscode_{category}", mode='clean'):
                paths = [os.path.join(dresscode_dataroot, category, 'images', line.strip().split()[0]) for line in lines if
                         os.path.exists(os.path.join(dresscode_dataroot, category, 'images', line.strip().split()[0]))]
                tmp_folder = f"/tmp/dresscode/{category}"
                os.makedirs(tmp_folder, exist_ok=True)
                for path in tqdm(paths):
                    shutil.copy(path, tmp_folder)
                fid.make_custom_stats(f"dresscode_{category}", tmp_folder, mode="clean", verbose=True)

        if not fid.test_stats_exists(f"dresscode_all", mode='clean'):
            paths = [os.path.join(dresscode_dataroot, category, 'images', line.strip().split()[0]) for line in lines for
                     category in ['lower_body', 'upper_body', 'dresses'] if
                     os.path.exists(os.path.join(dresscode_dataroot, category, 'images', line.strip().split()[0]))]
            tmp_folder = f"/tmp/dresscode/all"
            os.makedirs(tmp_folder, exist_ok=True)
            for path in tqdm(paths):
                shutil.copy(path, tmp_folder)
            fid.make_custom_stats(f"dresscode_all", tmp_folder, mode="clean", verbose=True)

    if vitonhd_dataroot is not None:
        if not fid.test_stats_exists(f"vitonhd_all", mode='clean'):
            fid.make_custom_stats(f"vitonhd_all", os.path.join(vitonhd_dataroot, 'test', 'image'), mode="clean",
                                  verbose=True)
        if not fid.test_stats_exists(f"vitonhd_upper_body", mode='clean'):
            fid.make_custom_stats(f"vitonhd_upper_body", os.path.join(vitonhd_dataroot, 'test', 'image'), mode="clean",
                                  verbose=True)
