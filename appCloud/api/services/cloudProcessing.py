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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
b_model = load_model(device)


def dnnInferenceCloud(feature, conf_list, p_tar):
	feature = torch.Tensor(feature).to(device)
	output, conf_list, inf_class = early_exit_dnn_inference(feature, conf_list, p_tar)

	return {"status": "ok"}



def early_exit_dnn_inference(tensor, conf_list, p_tar):
	
	b_model.eval()
	with torch.no_grad():
		output, conf_list, infer_class = b_model.forwardExperiment(tensor.float(), conf_list, p_tar=p_tar)
	return output, conf_list, infer_class
