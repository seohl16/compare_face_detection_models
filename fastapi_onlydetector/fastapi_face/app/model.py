import io
from typing import List, Dict, Any

import albumentations
import albumentations.pytorch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import MTCNN

def predict_from_image_byte(model: MTCNN, image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.array(image)
    bboxes, probs = model.detect(image_array)
    return bboxes

