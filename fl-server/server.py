"""
Main Flower server application for medical image segmentation
"""
import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import flwr as fl
from flwr.common import Metrics
from flwr.server import ServerConfig, ServerApp
from flwr.server.client_manager import SimpleClientManager
from flwr.server.history import History
from flwr.server.strategy import Strategy
from config.config import config
from strategies.fedavg_with_dp import FedAvgWithDP
from utils.logger import setup_logger
from utils.monitoring import MetricsCollector
from models.unet3d import create_unet3d_model
# Setup logger
logger = setup_logger(__name__, config.log_level)
class MedicalFLServerApp(ServerApp):
"""Custom Flower ServerApp for medical FL"""
def __init__(self):
self.metrics_collector = MetricsCollector()
self.experiment_id = None
def __call__(self, *args, **kwargs) -> Tuple[Optional[ServerConfig], Strategy]:
"""Configure server strategy"""
# Create initial model
model = create_unet3d_model(config.model_config)
# Get initial parameters
initial_parameters = fl.common.weights_to_parameters(
[param.detach().numpy() for param in model.parameters()]
)
# Create strategy
strategy = FedAvgWithDP(
fraction_fit=1.0,
fraction_eval=1.0,
min_fit_clients=config.min_clients,
min_eval_clients=config.min_clients,
min_available_clients=config.min_available_clients,
initial_parameters=initial_parameters,
on_fit_config_fn=self.fit_config,
on_evaluate_config_fn=self.evaluate_config,
)
# Server configuration
server_config = ServerConfig(
num_rounds=config.num_rounds,
round_timeout=config.round_timeout,
)
return server_config, strategy
def fit_config(self, rnd: int) -> Dict[str, str]:
"""Return fit configuration for each round"""
return {
'current_round': str(rnd),
'total_rounds': str(config.num_rounds),
'epochs': str(config.epochs_per_round),
'batch_size': str(config.batch_size),
'learning_rate': str(config.learning_rate),
'enable_dp': str(config.enable_dp).lower(),
'dp_epsilon': str(config.dp_epsilon),
'dp_delta': str(config.dp_delta),
'dp_max_grad_norm': str(config.dp_max_grad_norm),
}
def evaluate_config(self, rnd: int) -> Dict[str, str]:
"""Return evaluation configuration for each round"""
return {
'current_round': str(rnd),
'batch_size': str(config.batch_size),
}
def weighted_average(self, metrics: List[Tuple[int, Metrics]]) -> Metrics:
"""Aggregate metrics using weighted average"""
# Multiply accuracy of each client by number of examples used
accuracies = [num_examples * m['accuracy'] for num_examples, m in metrics]
examples = [num_examples for num_examples, _ in metrics]
# Aggregate and return custom metric (weighted average)
return {'accuracy': sum(accuracies) / sum(examples)}
def start_server(
server_address: str = config.server_address,
server_port: int = config.server_port,
experiment_id: Optional[str] = None,
) -> History:
"""
Start Flower server
Args:
server_address: Server address
server_port: Server port
experiment_id: Experiment ID for tracking
Returns:
Training history
"""
logger.info(f"Starting Medical FL Server on {server_address}:{server_port}")
if experiment_id:
logger.info(f"Experiment ID: {experiment_id}")
# Create server app
server_app = MedicalFLServerApp()
server_app.experiment_id = experiment_id
# Start server
try:
# Configure SSL if enabled
ssl_config = None
if config.enable_ssl:
ssl_config = (
config.certificate_path,
config.private_key_path,
config.ca_certificate_path,
)
logger.info("SSL enabled")
# Start Flower server
history = fl.server.start_server(
server_address=server_address,
server_port=server_port,
config=fl.server.ServerConfig(num_rounds=config.num_rounds),
strategy=server_app.strategy,
ssl_config=ssl_config,
)
logger.info("Server finished successfully")
return history
except Exception as e:
logger.error(f"Server error: {str(e)}")
raise
async def start_server_async(
server_address: str = config.server_address,
server_port: int = config.server_port,
) -> None:
"""Start server asynchronously"""
try:
history = await asyncio.to_thread(
start_server, server_address, server_port
)
return history
except Exception as e:
logger.error(f"Async server error: {str(e)}")
raise
def main():
"""Main entry point"""
# Parse command line arguments
import argparse
parser = argparse.ArgumentParser(description='Medical FL Server')
parser.add_argument(
'
--address',
type=str,
default=config.server_address,
help='Server address'
)
parser.add_argument(
'
--port',
type=int,
default=config.server_port,
help='Server port'
)
parser.add_argument(
'
--experiment-id',
type=str,
help='Experiment ID for tracking'
)
parser.add_argument(
'
--config',
type=str,
help='Path to configuration file'
)
args = parser.parse_args()
# Load custom config if provided
if args.config:
config.load_from_file(args.config)
try:
# Start server
history = start_server(
server_address=args.address,
server_port=args.port,
experiment_id=args.experiment_id,
)
# Print results
if history:
logger.info("Training completed")
losses_centralized = history.losses_centralized
metrics_centralized = history.metrics_centralized
if losses_centralized:
logger.info(f"Final loss: {losses_centralized[-1][1]}")
if metrics_centralized:
logger.info(f"Final metrics: {metrics_centralized}")
except KeyboardInterrupt:
logger.info("Server stopped by user")
sys.exit(0)
except Exception as e:
logger.error(f"Failed to start server: {str(e)}")
sys.exit(1)
if __name__ == "__main__":
main()
