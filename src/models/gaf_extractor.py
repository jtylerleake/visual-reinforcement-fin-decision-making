"""
Custom feature extractor for GAF (Gramian Angular Field) observations
"""

import os
import sys

import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class GAFExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for GAF observations.
    
    This extractor is designed to process multi-channel GAF images where each channel
    represents a different financial feature (Close, High, Low, Open, SMA, OBV).
    """
    
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        """
        Initialize the GAF feature extractor.
        
        Args:
            observation_space: The observation space (should be Box with shape (channels, height, width))
            features_dim: The dimension of the output features
        """
        super().__init__(observation_space, features_dim)
        
        # Extract dimensions from observation space
        channels = observation_space.shape[0]
        height = observation_space.shape[1]
        width = observation_space.shape[2]
        
        # Define CNN layers for GAF processing
        # The architecture is designed to handle small GAF images (typically 14x14)
        self.cnn = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(channels, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # Second convolutional layer
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # Third convolutional layer with pooling
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((4, 4)),  # Adaptive pooling to handle different input sizes
            
            # Flatten and fully connected layers
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, features_dim),
            nn.ReLU(),
            nn.Dropout(0.2)  # Add dropout for regularization
        )
    
    def forward(self, observations):
        """
        Forward pass through the feature extractor.
        
        Args:
            observations: Input observations of shape (batch_size, channels, height, width)
            
        Returns:
            Extracted features of shape (batch_size, features_dim)
        """
        return self.cnn(observations)
