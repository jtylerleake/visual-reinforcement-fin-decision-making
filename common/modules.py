"""Centralized module import location for project"""

# General Modules
import os
import sys
import io
import json
import importlib.util as imut
import threading
from pathlib import Path as path
import glob

# Typing Module Elements
from typing import List, Dict, Tuple, Optional, Union, Any

# Logging Modules
import logging
import logging.handlers as handlers
import traceback

# GUI Modules
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns

# Test Modules
import pytest
from unittest.mock import Mock, patch

# Time-Oriented Modules
import time
from datetime import datetime as dt
from datetime import timedelta as td
from pandas.tseries.holiday import USFederalHolidayCalendar

# Data Modules
import numpy as np # (version 1.16.5 required)
import pandas as pd
import random

# Finance modules
import yfinance as YF
from finta import TA
#import mplfinance as MPF

# Image Processing Modules
from pyts.image import GramianAngularField
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

# Machine Learning Modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as f

# Reinforcement Learning Modules
from gymnasium import spaces
import gym_anytrading
from gym_anytrading.envs import StocksEnv
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN, A2C, PPO
