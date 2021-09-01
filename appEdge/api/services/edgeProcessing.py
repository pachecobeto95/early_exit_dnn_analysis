from flask import jsonify, session, current_app as app
import os, pickle, requests, sys, config, time
import numpy as np, json
import torchvision.models as models
import torch
import datetime
import time, io
#import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
#from .utils import transform_image, load_model
from .utils import transform_image
from .utils import ModelLoad
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#b_model = load_model(device)
model = ModelLoad()

def edgeInference(fileImg, p_tar, nr_branch_edge):

	#This line reads the fileImg, obtaining pixel matrix.
	image_bytes = fileImg.read()
	response_request = {"status": "ok"}

	#Starts measuring the inference time
	start = time.time()
	tensor_img = transform_image(image_bytes) #transform input data, which means resize the input image

	#Run the Early-exit dnn inference
	output, conf_list, class_list, isTerminate = early_exit_dnn_inference_edge(tensor_img, p_tar, nr_branch_edge)

	if (not isTerminate):
		response_request = sendToCloud(output, conf_list, class_list, p_tar, nr_branch_edge)

	inference_time = time.time() - start
	if(response_request["status"] == "ok"):
		saveInferenceTime(inference_time,  p_tar, nr_branch_edge, model.nr_branches_model)
	
	return response_request


def sendToCloud(feature_map, conf_list, class_list, p_tar, nr_branch_edge):
	"""
	This functions sends output data from a partitioning layer from edge device to cloud server.
	This function also sends the info of partitioning layer to the cloud.
	Argments:

	feature_map (Tensor): output data from partitioning layer
	conf_list (list): this list contains the confidence obtained for each early exit during Early-exit DNN inference
	"""
	
	data = {'feature': feature_map.detach().cpu().numpy().tolist(), "conf": conf_list, "p_tar": p_tar, 
	"nr_branch_edge": nr_branch_edge, "class_list": class_list}

	try:
		response = requests.post(config.URL_CLOUD_DNN_INFERENCE, json=data, timeout=config.timeout)
		response.raise_for_status()
		return {"status": "ok"}

	except Exception as e:
		print(e)
		return {"status": "error"}


def early_exit_dnn_inference_edge(tensor_img, p_tar, nr_branch_edge):
	model.b_model.eval()

	with torch.no_grad():
		output, conf_list, class_list, isTerminate = model.b_model.forwardEdgeInference(tensor_img.to(device).float(), p_tar, 
			nr_branch_edge)

	return output, conf_list, class_list, isTerminate

def saveInferenceTime(inference_time,  p_tar, nr_branch_edge, nr_branches_model):
	
	result = {"inference_time": inference_time, "p_tar": p_tar, "nr_branch_edge": nr_branch_edge, 
	"nr_branch_model": nr_branches_model}

	result_path = os.path.join(config.RESULTS_INFERENCE_TIME_EDGE, "inference_time.csv")

	if (not os.path.exists(result_path)):
		df = pd.DataFrame()
	else:
		df = pd.read_csv(result_path)
		df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
	
	df = df.append(pd.Series(result), ignore_index=True)
	df.to_csv(result_path) 
