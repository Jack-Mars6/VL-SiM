import glob
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from monai.data import PersistentDataset, DataLoader
from monai.transforms import Compose, Transform
import nibabel as nib
import torch.nn.functional as F
import torchvision.transforms as transforms
from functools import partial


class CTPreprocessTransform(Transform):

    def __init__(self, target_shape=(224, 224, 160), target_spacing=(2.0, 1.5, 1.5)):
        self.target_shape = target_shape
        self.target_spacing = target_spacing
        self.hu_min, self.hu_max = -1000, 1000

    def __call__(self, data):
        nii_path = data["image"]
        metadata = data["metadata"]
        nii_img = nib.load(str(nii_path))
        img_data = nii_img.get_fdata()
        slope = metadata['slope']
        intercept = metadata['intercept']
        img_data = slope * img_data + intercept
        processed_tensor = self._process_volume(img_data, metadata)

        return {
            "image": processed_tensor,
            "text": data["text"],
            "image_path": str(nii_path)
        }

    def _process_volume(self, img_data, metadata):
        img_data = img_data.transpose(2, 0, 1)  # xyz -> zxy

        tensor = torch.tensor(img_data).unsqueeze(0).unsqueeze(0)
        current_spacing = (metadata['z_spacing'], metadata['xy_spacing'], metadata['xy_spacing'])
        resized_data = self._resample_volume(tensor, current_spacing, self.target_spacing)
        resized_data = np.transpose(resized_data, (1, 2, 0))  # zxy -> xyz

        resized_data = np.clip(resized_data, self.hu_min, self.hu_max)
        resized_data = ((resized_data - self.hu_min) / 2000).astype(np.float32)
        # print(resized_data.shape)

        return self._crop_and_pad(torch.tensor(resized_data))

    def _resample_volume(self, volume, current_spacing, target_spacing):
        scaling_factors = [
            current_spacing[i] / target_spacing[i]
            for i in range(len(current_spacing))
        ]

        original_shape = volume.shape[2:]
        new_shape = [
            int(original_shape[i] * scaling_factors[i])
            for i in range(len(original_shape))
        ]

        resized_volume = F.interpolate(
            volume, size=new_shape, mode='trilinear', align_corners=False
        )
        return resized_volume.squeeze(0).squeeze(0).numpy()

    def _crop_and_pad(self, tensor):
        h, w, d = tensor.shape
        dh, dw, dd = self.target_shape

        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before

        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before

        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # HWD -> CDHW
        return tensor


class CTPersistentDataset:

    def __init__(self, data_folder, csv_file, cache_dir="./ct_cache",
                 target_shape=(224, 224, 160)):

        self.data_folder = data_folder

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_cache = self._load_metadata_cache()
        self.accession_to_text = self._load_accession_text(csv_file)
        self.data_dicts = self._prepare_data_dicts()

        # # 测试
        # percent = 0.5
        # num_files = int((len(self.data_dicts) * percent) / 100)
        # self.data_dicts = self.data_dicts[:num_files]

        self.preprocess_transform = CTPreprocessTransform(target_shape=target_shape)
        self.dataset = PersistentDataset(
            data=self.data_dicts,
            transform=self.preprocess_transform,
            cache_dir=self.cache_dir,
            pickle_protocol=4
        )

    def _load_metadata_cache(self):
        metadata_path = "/home2/Reports_label/metadata/train_metadata.csv"
        df = pd.read_csv(metadata_path)
        metadata_cache = {}

        for _, row in df.iterrows():
            volume_name = row['VolumeName']
            xy_spacing = float(row["XYSpacing"][1:][:-2].split(",")[0])
            metadata_cache[volume_name] = {
                'slope': float(row["RescaleSlope"]),
                'intercept': float(row["RescaleIntercept"]),
                'xy_spacing': xy_spacing,
                'z_spacing': float(row["ZSpacing"])
            }
        return metadata_cache

    def _load_accession_text(self, csv_file):
        df = pd.read_csv(csv_file)
        accession_to_text = {}
        for _, row in df.iterrows():
            findings = str(row["Findings_EN"]) if pd.notna(row["Findings_EN"]) else ""
            impressions = str(row['Impressions_EN']) if pd.notna(row['Impressions_EN']) else ""

            combined_text = f"{findings} {impressions}".strip()
            combined_text = combined_text.replace('"', '').replace('\'', '').replace('(', '').replace(')', '')
            if combined_text == "Not given.":
                combined_text = ""

            accession_to_text[row['VolumeName']] = combined_text
        return accession_to_text

    def _prepare_data_dicts(self):

        data_dicts = []
        data_folders = self.data_folder
        for data_folder in data_folders:
            for patient_folder in sorted(glob.glob(os.path.join(data_folder, '*'))):
                for accession_folder in glob.glob(os.path.join(patient_folder, '*')):

                    for nii_file in glob.glob(os.path.join(accession_folder, '*.nii.gz')):
                        accession_number = nii_file.split("/")[-1]

                        if (accession_number not in self.metadata_cache or
                                accession_number not in self.accession_to_text):
                            continue

                        text = self.accession_to_text[accession_number]
                        metadata = self.metadata_cache[accession_number]

                        data_dicts.append({
                            "image": str(nii_file),
                            "text": text,
                            "metadata": metadata,
                            "accession_number": accession_number
                        })

        return data_dicts


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            data_item = self.dataset[index]
            return data_item["image"], data_item["text"]

        except Exception as e:
            print(f"process {index} error：{e}")
            next_index = (index + 1) % len(self.dataset)
            return self.__getitem__(next_index)
