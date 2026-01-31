"""
Custom Flower strategy with differential privacy
"""
import copy
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from opacus import PrivacyEngine
import flwr as fl
from flwr.common import (
EvaluateRes,
FitRes,
Parameters,
Scalar,
Weights,
parameters_to_weights,
weights_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from ..config.config import config
class FedAvgWithDP(FedAvg):
"""FedAvg strategy with differential privacy support"""
def __init__(
self,
fraction_fit: float = 1.0,
fraction_eval: float = 1.0,
min_fit_clients: int = 2,
min_eval_clients: int = 2,
min_available_clients: int = 2,
eval_fn=None,
on_fit_config_fn=None,
on_evaluate_config_fn=None,
accept_failures: bool = True,
initial_parameters=None,
privacy_engine=None,
) -> None:
super().__init__(
fraction_fit=fraction_fit,
fraction_eval=fraction_eval,
min_fit_clients=min_fit_clients,
min_eval_clients=min_eval_clients,
min_available_clients=min_available_clients,
eval_fn=eval_fn,
on_fit_config_fn=on_fit_config_fn,
on_evaluate_config_fn=on_evaluate_config_fn,
accept_failures=accept_failures,
initial_parameters=initial_parameters,
)
self.privacy_engine = privacy_engine
self.privacy_budget_used = 0.0
def aggregate_fit(
self,
rnd: int,
results: List[Tuple[ClientProxy, FitRes]],
failures: List[BaseException],
) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
"""Aggregate fit results with differential privacy"""
if not results:
return None, {}
# Convert results
weights_results = [
(parameters_to_weights(fit_res.parameters), fit_res.num_examples)
for _, fit_res in results
]
# Apply differential privacy if enabled
if config.enable_dp and self.privacy_engine:
weights_results = self._apply_differential_privacy(weights_results, rnd)
# Aggregate weights using weighted average
aggregated_weights = self._weighted_average(weights_results)
# Convert weights to parameters
aggregated_parameters = weights_to_parameters(aggregated_weights)
# Track privacy budget
if config.enable_dp:
self.privacy_budget_used += config.dp_epsilon / config.num_rounds
# Aggregate custom metrics if available
metrics_aggregated = {}
if results:
metrics_aggregated = self._aggregate_metrics(results)
return aggregated_parameters, metrics_aggregated
def _apply_differential_privacy(
self,
weights_results: List[Tuple[Weights, int]],
rnd: int,
) -> List[Tuple[Weights, int]]:
"""Apply differential privacy to weights"""
noisy_weights_results = []
for weights, num_examples in weights_results:
# Add Gaussian noise to weights
noisy_weights = []
for layer_weights in weights:
# Calculate noise scale based on privacy budget
noise_scale = config.dp_max_grad_norm * np.sqrt(
2 * np.log(1.25 / config.dp_delta)
) / config.dp_epsilon
# Add noise
noise = np.random.normal(0, noise_scale, layer_weights.shape)
noisy_layer = layer_weights + noise
noisy_weights.append(noisy_layer)
noisy_weights_results.append((noisy_weights, num_examples))
return noisy_weights_results
def _weighted_average(
self, weights_results: List[Tuple[Weights, int]]
) -> Weights:
"""Compute weighted average of weights"""
# Calculate total examples
total_examples = sum(num_examples for _, num_examples in weights_results)
# Initialize aggregated weights
aggregated_weights = []
num_layers = len(weights_results[0][0])
for layer_idx in range(num_layers):
# Weighted average for each layer
weighted_sum = np.zeros_like(weights_results[0][0][layer_idx])
for weights, num_examples in weights_results:
layer_weight = weights[layer_idx]
weighted_sum += layer_weight * num_examples
aggregated_layer = weighted_sum / total_examples
aggregated_weights.append(aggregated_layer)
return aggregated_weights
def _aggregate_metrics(
self, results: List[Tuple[ClientProxy, FitRes]]
) -> Dict[str, Scalar]:
"""Aggregate metrics from clients"""
metrics_aggregated = {}
# Aggregate loss and accuracy
losses = []
accuracies = []
num_examples = []
for _, fit_res in results:
metrics = fit_res.metrics
if metrics:
if 'loss' in metrics:
losses.append(metrics['loss'])
if 'accuracy' in metrics:
accuracies.append(metrics['accuracy'])
num_examples.append(fit_res.num_examples)
if losses:
# Weighted average of losses
total_examples = sum(num_examples)
weighted_loss = sum(
loss * num for loss, num in zip(losses, num_examples)
) / total_examples
metrics_aggregated['loss'] = weighted_loss
if accuracies:
# Weighted average of accuracies
weighted_accuracy = sum(
acc * num for acc, num in zip(accuracies, num_examples)
) / total_examples
metrics_aggregated['accuracy'] = weighted_accuracy
# Add privacy budget information
if config.enable_dp:
metrics_aggregated['privacy_budget_used'] = self.privacy_budget_used
metrics_aggregated['epsilon_remaining'] = (
config.dp_epsilon - self.privacy_budget_used
)
return metrics_aggregated
def configure_fit(
self, rnd: int, parameters: Parameters, client_manager
) -> List[Tuple[ClientProxy, Dict[str, Scalar]]]:
"""Configure the next round of training"""
# Get standard configuration
config = super().configure_fit(rnd, parameters, client_manager)
# Add custom configuration
for _, fit_config in config:
fit_config['current_round'] = rnd
fit_config['total_rounds'] = self.min_fit_clients
fit_config['epochs'] = config.epochs_per_round
fit_config['batch_size'] = config.batch_size
fit_config['learning_rate'] = config.learning_rate
if self.privacy_engine:
fit_config['enable_dp'] = True
fit_config['dp_epsilon'] = config.dp_epsilon
fit_config['dp_delta'] = config.dp_delta
fit_config['dp_max_grad_norm'] = config.dp_max_grad_norm
return config
def configure_evaluate(
self, rnd: int, parameters: Parameters, client_manager
) -> List[Tuple[ClientProxy, Dict[str, Scalar]]]:
"""Configure the next round of evaluation"""
config = super().configure_evaluate(rnd, parameters, client_manager)
for _, eval_config in config:
eval_config['current_round'] = rnd
return config
