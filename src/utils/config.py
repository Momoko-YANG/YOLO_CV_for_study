# -*- coding: utf-8 -*-
"""
Configuration management for the gesture recognition system.
Centralizes all configurable parameters in one place.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import torch


@dataclass
class ModelConfig:
    """Configuration for YOLO model parameters."""
    
    # Device configuration
    device: str = field(default_factory=lambda: "cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Detection thresholds
    conf_threshold: float = 0.25
    iou_threshold: float = 0.5
    
    # Image processing
    image_size: int = 640
    
    # Model paths
    default_model_path: str = "weights/best-yolov8n.pt"
    
    # Class filtering (None means no filtering)
    classes: Optional[list] = None
    
    # Verbose output
    verbose: bool = False
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for model inference."""
        return {
            'device': self.device,
            'conf': self.conf_threshold,
            'iou': self.iou_threshold,
            'classes': self.classes,
            'verbose': self.verbose
        }


@dataclass
class UIConfig:
    """Configuration for UI parameters."""
    
    # Sidebar dimensions
    sidebar_collapsed_width: int = 55
    sidebar_expanded_width: int = 240
    
    # Animation
    animation_duration: int = 400
    
    # Media processing
    default_fps: int = 30
    default_camera_id: int = 0
    
    # Table column widths
    table_column_widths: Tuple[int, ...] = (80, 200, 150, 200, 120)
    
    # Window control buttons
    button_sizes: Tuple[int, int] = (20, 20)
    button_gaps: int = 30
    button_right_margin: int = 50
    
    # Default frame counts
    default_total_frames: int = 1000
    
    # Video output settings
    video_output_duration: int = 10  # seconds


@dataclass
class AppConfig:
    """Main application configuration."""
    
    # Application info
    app_name: str = "Gesture Recognition System"
    app_version: str = "1.0.0"
    
    # Language settings
    language: str = "en"  # "en" or "zh"
    
    # Database
    database_path: str = "UserDatabase.db"
    
    # Theme paths
    main_theme_yaml: str = "themes/Settings_main.yaml"
    main_theme_qss: str = "themes/main_text_black.qss"
    login_theme_yaml: str = "themes/Settings_login.yaml"
    login_theme_qss: str = "themes/login_text_black.qss"
    
    # Output directories
    output_dir: str = "./output_examples"
    
    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    @classmethod
    def load_default(cls) -> 'AppConfig':
        """Load default configuration."""
        return cls()


# Global configuration instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = AppConfig.load_default()
    return _config


def set_config(config: AppConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
