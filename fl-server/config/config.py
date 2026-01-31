"""
Flower server configuration
"""
import os
from typing import Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv
load_dotenv()
@dataclass
class ServerConfig:
"""Server configuration"""
# Server settings
server_address: str = os.getenv('FLOWER_SERVER_HOST', '0.0.0.0')
server_port: int = int(os.getenv('FLOWER_SERVER_PORT', 8080))
num_rounds: int = int(os.getenv('FL_ROUNDS', 50))
round_timeout: int = 600 # seconds
# Security
enable_ssl: bool = os.getenv('FLOWER_SSL_ENABLED', 'false').lower() == 'true'
certificate_path: str = os.getenv('FLOWER_CERT_PATH', '/certs/server.crt')
private_key_path: str = os.getenv('FLOWER_KEY_PATH', '/certs/server.key')
ca_certificate_path: str = os.getenv('FLOWER_CA_PATH', '/certs/ca.crt')
# Client management
min_clients: int = int(os.getenv('FL_MIN_CLIENTS', 3))
min_available_clients: int = int(os.getenv('FL_MIN_AVAILABLE_CLIENTS', 5))
max_clients: int = 100
client_keepalive: int = 60 # seconds
# Privacy
enable_dp: bool = True
dp_epsilon: float = float(os.getenv('DP_EPSILON', 1.0))
dp_delta: float = float(os.getenv('DP_DELTA', 1e-5))
dp_max_grad_norm: float = float(os.getenv('DP_MAX_GRAD_NORM', 1.0))
# Model settings
model_type: str = os.getenv('MODEL_TYPE', 'unet3d')
model_config: Dict[str, Any] = field(default_factory=lambda: {
'in_channels': int(os.getenv('MODEL_IN_CHANNELS', 1)),
'out_channels': int(os.getenv('MODEL_OUT_CHANNELS', 2)),
'features': eval(os.getenv('MODEL_FEATURES', '[32, 64, 128, 256]')),
'dropout': float(os.getenv('MODULE_DROPOUT', 0.2)),
})
# Training settings
epochs_per_round: int = int(os.getenv('FL_EPOCHS_PER_ROUND', 2))
batch_size: int = int(os.getenv('FL_BATCH_SIZE', 8))
learning_rate: float = float(os.getenv('FL_LEARNING_RATE', 0.001))
# Storage
model_save_path: str = os.getenv('MODEL_SAVE_PATH', './saved_models')
checkpoint_path: str = os.getenv('CHECKPOINT_PATH', './checkpoints')
# Monitoring
enable_monitoring: bool = True
metrics_port: int = 9091
log_level: str = os.getenv('LOG_LEVEL', 'INFO')
def to_dict(self) -> Dict[str, Any]:
"""Convert to dictionary"""
return {
'server_address': self.server_address,
'server_port': self.server_port,
'num_rounds': self.num_rounds,
'min_clients': self.min_clients,
'enable_ssl': self.enable_ssl,
'enable_dp': self.enable_dp,
'dp_epsilon': self.dp_epsilon,
'model_type': self.model_type,
'epochs_per_round': self.epochs_per_round,
'batch_size': self.batch_size,
'learning_rate': self.learning_rate,
}
# Global configuration instance
config = ServerConfig()
