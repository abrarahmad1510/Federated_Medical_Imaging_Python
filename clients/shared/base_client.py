"""
Base Flower client for medical image segmentation
"""
import copy
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import (
EvaluateIns,
EvaluateRes,
FitIns,
FitRes,
Parameters,
Scalar,
Weights,
parameters_to_weights,
weights_to_parameters,
)
import numpy as np
from .data_loader import MedicalDataset
from .models import create_model
logger = logging.getLogger(__name__)
class MedicalFLClient(fl.client.Client):
"""Base Flower client for medical FL"""
def __init__(
self,
client_id: str,
data_dir: Path,
model_config: Dict,
device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
"""
Initialize client
Args:
client_id: Unique client identifier
data_dir: Directory containing client data
model_config: Model configuration
device: Device for training (cuda/cpu)
"""
self.client_id = client_id
self.data_dir = Path(data_dir)
self.device = device
self.model_config = model_config
# Initialize model, optimizer, criterion
self.model = None
self.optimizer = None
self.criterion = None
# Initialize datasets
self.train_dataset = None
self.val_dataset = None
self.test_dataset = None
# Training statistics
self.train_stats = {
'total_samples': 0,
'total_batches': 0,
'total_epochs': 0,
}
# Privacy settings
self.enable_dp = False
self.privacy_engine = None
logger.info(f"Client {client_id} initialized on device: {device}")
def initialize_model(self) -> None:
"""Initialize model, optimizer, and criterion"""
# Create model
self.model = create_model(self.model_config)
self.model.to(self.device)
# Create optimizer
self.optimizer = optim.Adam(
self.model.parameters(),
lr=0.001, # Will be overridden by server config
weight_decay=1e-4,
)
# Create criterion (Dice + CrossEntropy for segmentation)
self.criterion = self._create_criterion()
logger.info(f"Model initialized for client {self.client_id}")
def _create_criterion(self) -> nn.Module:
"""Create loss criterion for medical segmentation"""
from monai.losses import DiceCELoss
return DiceCELoss(
include_background=True,
to_onehot_y=True,
softmax=True,
squared_pred=True,
smooth_nr=1e-5,
smooth_dr=1e-5,
)
def load_data(self) -> None:
"""Load client data"""
# This should be implemented by subclasses
raise NotImplementedError("Subclasses must implement load_data")
def get_train_loader(self, batch_size: int) -> DataLoader:
"""Get training DataLoader"""
if self.train_dataset is None:
self.load_data()
return DataLoader(
self.train_dataset,
batch_size=batch_size,
shuffle=True,
num_workers=2,
pin_memory=True,
drop_last=True,
)
def get_val_loader(self, batch_size: int) -> DataLoader:
"""Get validation DataLoader"""
if self.val_dataset is None:
self.load_data()
return DataLoader(
self.val_dataset,
batch_size=batch_size,
shuffle=False,
num_workers=2,
pin_memory=True,
)
def get_test_loader(self, batch_size: int) -> DataLoader:
"""Get test DataLoader"""
if self.test_dataset is None:
self.load_data()
return DataLoader(
self.test_dataset,
batch_size=batch_size,
shuffle=False,
num_workers=2,
pin_memory=True,
)
def train_epoch(
self,
train_loader: DataLoader,
epoch: int,
max_batches: Optional[int] = None,
) -> Dict[str, float]:
"""
Train for one epoch
Returns:
Dictionary with training metrics
"""
self.model.train()
total_loss = 0.0
total_samples = 0
for batch_idx, batch in enumerate(train_loader):
if max_batches and batch_idx >= max_batches:
break
# Move data to device
images = batch['image'].to(self.device)
masks = batch['mask'].to(self.device)
# Zero gradients
self.optimizer.zero_grad()
# Forward pass
outputs = self.model(images)
# Calculate loss
loss = self.criterion(outputs, masks)
# Backward pass
loss.backward()
# Clip gradients if DP enabled
if self.enable_dp and self.privacy_engine:
self.privacy_engine.clip_gradients()
# Optimizer step
self.optimizer.step()
# Update statistics
batch_size = images.size(0)
total_loss += loss.item() * batch_size
total_samples += batch_size
# Log progress
if batch_idx % 10 == 0:
logger.debug(
f"Client {self.client_id} - Epoch {epoch} "
f"[{batch_idx}/{len(train_loader)}] "
f"Loss: {loss.item():.4f}"
)
# Calculate average loss
avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
# Update training statistics
self.train_stats['total_samples'] += total_samples
self.train_stats['total_batches'] += len(train_loader)
self.train_stats['total_epochs'] += 1
return {
'loss': avg_loss,
'samples': total_samples,
'batches': len(train_loader),
}
def evaluate(
self,
data_loader: DataLoader,
max_batches: Optional[int] = None,
) -> Dict[str, float]:
"""
Evaluate model on data
Returns:
Dictionary with evaluation metrics
"""
self.model.eval()
total_loss = 0.0
total_dice = 0.0
total_samples = 0
with torch.no_grad():
for batch_idx, batch in enumerate(data_loader):
if max_batches and batch_idx >= max_batches:
break
# Move data to device
images = batch['image'].to(self.device)
masks = batch['mask'].to(self.device)
# Forward pass
outputs = self.model(images)
# Calculate loss
loss = self.criterion(outputs, masks)
# Calculate Dice score
dice_score = self._calculate_dice_score(outputs, masks)
# Update statistics
batch_size = images.size(0)
total_loss += loss.item() * batch_size
total_dice += dice_score * batch_size
total_samples += batch_size
# Calculate averages
avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
avg_dice = total_dice / total_samples if total_samples > 0 else 0.0
return {
'loss': avg_loss,
'dice_score': avg_dice,
'samples': total_samples,
'accuracy': avg_dice, # Using dice as accuracy proxy
}
def _calculate_dice_score(
self,
predictions: torch.Tensor,
targets: torch.Tensor,
smooth: float = 1e-5,
) -> float:
"""Calculate Dice score for segmentation"""
# Convert to binary predictions
preds = torch.argmax(predictions, dim=1)
# Calculate intersection and union
intersection = (preds * targets).sum()
union = preds.sum() + targets.sum()
# Calculate Dice
dice = (2.0 * intersection + smooth) / (union + smooth)
return dice.item()
def get_parameters(self) -> List[np.ndarray]:
"""Get model parameters as list of NumPy ndarrays"""
return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
def set_parameters(self, parameters: List[np.ndarray]) -> None:
"""Set model parameters from list of NumPy ndarrays"""
params_dict = zip(self.model.state_dict().keys(), parameters)
state_dict = {
k: torch.tensor(v) if isinstance(v, np.ndarray) else v
for k, v in params_dict
}
self.model.load_state_dict(state_dict, strict=True)
def enable_differential_privacy(
self,
epsilon: float,
delta: float,
max_grad_norm: float,
) -> None:
"""Enable differential privacy for this client"""
if not self.enable_dp:
try:
from opacus import PrivacyEngine
self.privacy_engine = PrivacyEngine()
self.model, self.optimizer, self.train_loader = (
self.privacy_engine.make_private(
module=self.model,
optimizer=self.optimizer,
data_loader=self.get_train_loader(batch_size=8),
noise_multiplier=1.1, # Will be calculated based on epsilon
max_grad_norm=max_grad_norm,
)
)
self.enable_dp = True
logger.info(
f"Client {self.client_id} - DP enabled "
f"(ε={epsilon}, δ={delta}, max_grad_norm={max_grad_norm})"
)
except ImportError:
logger.warning("Opacus not installed, DP disabled")
self.enable_dp = False
def get_client_info(self) -> Dict[str, Scalar]:
"""Get client information"""
return {
'client_id': self.client_id,
'device': self.device,
'train_samples': self.train_stats['total_samples'],
'total_epochs': self.train_stats['total_epochs'],
'enable_dp': self.enable_dp,
}
