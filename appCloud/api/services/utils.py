import os, pickle, requests, sys, config, time, json, io
import numpy as np
import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from .early_exit_dnn import Early_Exit_DNN
import pandas as pd

def transform_image(image_bytes):
	imagenet_mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
	imagenet_std = [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]
	my_transforms = transforms.Compose([transforms.Resize(config.input_dim),
		#transforms.CenterCrop(config.input_shape),
		transforms.ToTensor(),
		transforms.Normalize(imagenet_mean, imagenet_std)])

	image = Image.open(io.BytesIO(image_bytes))
	return my_transforms(image).unsqueeze(0)


def load_model(device):

	early_exit_model = Early_Exit_DNN(config.model_name, config.n_classes, config.pretrained, config.nr_branch_model, 
		config.input_shape, config.exit_type, device, distribution=config.distribution)


	early_exit_model.load_state_dict(torch.load(config.model_path_edge_final, map_location=device)["model_state_dict"])	
	return early_exit_model.to(device)

class ModelLoad():
	def __init__(self):
		self._nr_branches_model = None

	@property
	def nr_branches_model(self):
		return self._nr_branches_model


	@nr_branches_model.setter
	def nr_branches_model(self, nr_branches):
		self._nr_branches_model = nr_branches


	def update_load_model(self, device, dataset_name):

		self.b_model = Early_Exit_DNN(config.model_name, config.n_classes, config.pretrained, self.nr_branches_model, 
			config.input_shape, config.exit_type, device, distribution=config.distribution)

		model_path = os.path.join(config.cloud_model_path, "mobilenet_%s_2_branches_%s.pth"%(dataset_name, self.nr_branches_model))
		self.b_model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])	

	def load_temperature(self):
		temp_path = os.path.join(config.temp_cloud_path, "branches_temp_scaling_branches_%s_2.csv"%(self._nr_branches_model))
		df_temp = pd.read_csv(temp_path)
		df_temp = df_temp.loc[:, ~df_temp.columns.str.contains('^Unnamed')]
		df_temp = df_temp[df_temp.p_tar==0.8]
		temp_list = [nn.Parameter(torch.tensor([df_temp["temperature_branch_%s"%(i)].unique().item()])) for i in range(1, self._nr_branches_model+1)]
		self.b_model.temperature_branches = temp_list
