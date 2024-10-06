# Test instantiation outside the training loop
from dataset import DroneDataset
import torch

train_dataset = DroneDataset(
    "/workspaces/AE4353-Y24/data/AutonomousFlightData/test"
)  # Use a valid path
print(f"Loaded dataset with {len(train_dataset)} images.")

torch.cuda.empty_cache()
