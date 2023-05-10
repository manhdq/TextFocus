import os
import json
from shutil import rmtree
from datetime import datetime
import pickle

import yaml
import torch
import onnxruntime as rt

from utils import FocusChip
from utils.misc import remove_prefix
from utils.parser import infer_parser
from retinafocus import RetinaFocus, RetinaFocusWrapper
from data import InferenceDataloader, FocusGenerator
from inference import CountFaceInference