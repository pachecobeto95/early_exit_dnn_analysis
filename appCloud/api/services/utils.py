import os, pickle, requests, sys, config, time, json, io
import numpy as np
import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from .mobilenet import B_MobileNet

def load_model(device):
	b_mobilenet_model = B_MobileNet(config.n_classes, config.pretrained, config.n_branches, config.input_shape, 
		config.exit_type, device)

	b_mobilenet_model.load_state_dict(torch.load(config.model_path, map_location=device)["model_state_dict"])	
	return b_mobilenet_model.to(device)