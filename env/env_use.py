# -*- coding:utf-8  -*-
import os
import time
import json
import numpy as np
import argparse
import sys

sys.path.append("./olympics_engine")
sys.path.append("../")

from chooseenv import make
from utils.get_logger import get_logger
from obs_interfaces.observation import obs_type


env_type = "kafang_stock"
print("begin")
game = make(env_type, seed=None)

game.reset()