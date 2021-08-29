from flask import jsonify, session, current_app as app
import os, pickle, requests, sys, config, time
import numpy as np, json
import torchvision.models as models
import torch
import datetime
import time, io
import torchvision.transforms as transforms
from PIL import Image
from .utils import load_model
from .utils import ModelLoad


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#b_model = load_model(device)
model = ModelLoad()


def dnnInferenceCloud(feature, conf_list, class_list, p_tar, nr_branch_edge):
	feature = torch.Tensor(feature).to(device)
	conf, infer_class = early_exit_dnn_inference_cloud(feature, conf_list, class_list, p_tar, nr_branch_edge)

	return {"status": "ok"}



def early_exit_dnn_inference_cloud(x, conf_list, class_list, p_tar, nr_branch_edge):
	
	model.b_model.eval()
	with torch.no_grad():
		conf, infer_class = model.b_model.forwardCloudInference(x.float(), conf_list, class_list, p_tar, nr_branch_edge)
	return conf, infer_class
