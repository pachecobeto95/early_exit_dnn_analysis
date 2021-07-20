import os, pickle, requests, sys, config, time, json, io
import numpy as np
import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from .mobilenet import B_MobileNet

def transform_image(image_bytes):
	imagenet_mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
	imagenet_std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]
	my_transforms = transforms.Compose([transforms.Resize(config.resize_shape),
		transforms.CenterCrop(config.input_shape),
		transforms.ToTensor(),
		transforms.Normalize(imagenet_mean, imagenet_std)])

	image = Image.open(io.BytesIO(image_bytes))
	return my_transforms(image).unsqueeze(0)



def load_model(device):
	b_mobilenet_model = B_MobileNet(config.n_classes, config.pretrained, config.n_branches, config.input_shape, 
		config.exit_type, device)

	b_mobilenet_model.load_state_dict(torch.load(config.model_path, map_location=device)["model_state_dict"])	
	return b_mobilenet_model.to(device)