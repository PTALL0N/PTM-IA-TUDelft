import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import os
import hdf5plugin

## THINGS TO IMPROVE
# - To start off, the dataset used will be based on the usage of one set of samples .h5
# - Shuffle between the indexes of different .h5 files to reduce overfitting
# - Train on diffrent batches, start with a batch, then eliminate the dataset, add another set and continue training
# - ...


class DroneDataset(Dataset):
    def __init__(self, dir_dataset, max_gates=10):
        self.dataset_path = dir_dataset
        self.images_list = []
        self.targets_list = []
        self.max_gates = max_gates  # Maximum number of gates per image

        # Load the dataset
        self._load_all_h5_files()

    def _load_h5_file(self, h5_file_path):
        with h5py.File(h5_file_path, "r") as f:
            # Load images
            images = np.array(f["images"])
            print(f"Loaded {images.shape[0]} images from {h5_file_path}")

            # Load targets from the 'targets' group
            targets = []
            target_group = f["targets"]
            for target_name in target_group.keys():
                target_data = np.array(target_group[target_name])
                targets.append(torch.from_numpy(target_data).float())

            return images, targets

    def _load_all_h5_files(self):
        """Load images and targets from all .h5 files in the dataset path."""
        for filename in os.listdir(self.dataset_path):
            if filename.endswith(".h5"):
                h5_file_path = os.path.join(self.dataset_path, filename)
                images, targets = self._load_h5_file(h5_file_path)

                if images is not None and targets is not None:
                    self.images_list.append(torch.from_numpy(images).float())
                    padded_targets = [
                        self.pad_or_truncate_targets(t, self.max_gates) for t in targets
                    ]
                    self.targets_list.append(
                        torch.stack(padded_targets)
                    )  # Stack the padded tensors

        # Concatenate all images and targets
        self.images = (
            torch.cat(self.images_list, dim=0) if self.images_list else torch.empty(0)
        )
        self.targets = (
            torch.cat(self.targets_list, dim=0) if self.targets_list else torch.empty(0)
        )
        print(f"Total images loaded: {self.images.shape[0]}")
        print(f"Total targets loaded: {self.targets.shape[0]}")

    def pad_or_truncate_targets(self, target, max_gates):
        """Pads or truncates the target tensor to ensure it has `max_gates` rows."""
        num_gates, num_features = target.shape
        if num_gates < max_gates:
            # If fewer gates, pad with zeros
            padding = torch.zeros((max_gates - num_gates, num_features))
            target = torch.cat([target, padding], dim=0)
        elif num_gates > max_gates:
            # If more gates, truncate the target
            target = target[:max_gates, :]
        return target

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Return a single image and its corresponding padded target
        return self.images[idx], self.targets[idx]


# Example usage
# if __name__ == "__main__":
#     dataset = DroneDataset("/workspaces/AE4353-Y24/data/AutonomousFlightData/test")

#     # Access the combined images and targets
#     images, targets = dataset[0]

#     # Example: Print the shape of images and targets
#     print(f"Images shape: {images.shape}")
#     print(
#         f"Targets shape: {targets.shape}"
#     )  # Shape might vary per image depending on the number of gates
