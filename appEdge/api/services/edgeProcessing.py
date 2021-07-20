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
from .utils import transform_image, load_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
b_model = load_model(device)

def edgeInference(fileImg, p_tar):

	#This line reads the fileImg, obtaining pixel matrix.
	image_bytes = fileImg.read()

	#Starts measuring the inference time
	start = time.time()
	tensor_img = transform_image(image_bytes) #transform input data, which means resize the input image

	#Run the Early-exit dnn inference
	output, conf_list, infer_class = b_mobileNetInferenceEdge(tensor_img, p_tar)

	print("oi")
	response_request = {"status": "ok"}

	print(infer_class is None)
	#If infer_class is None, it indicates the inferece can't be done in the edge, so we send to the cloud terminates it.
	if (infer_class is None):
		response_request = sendToCloud(output, conf_list, p_tar)

	inference_time = time.time() - start
	#if (response_request["status"] == "ok"):
	#	saveInferenceTime(inference_time, p_tar)

	return response_request


def sendToCloud(feature_map, conf_list, p_tar):
	"""
	This functions sends output data from a partitioning layer from edge device to cloud server.
	This function also sends the info of partitioning layer to the cloud.
	Argments:

	feature_map (Tensor): output data from partitioning layer
	conf_list (list): this list contains the confidence obtained for each early exit during Early-exit DNN inference
	"""
	print("k")
	data = {'feature': feature_map.detach().cpu().numpy().tolist(), "conf": conf_list, "p_tar": p_tar}

	try:
		response = requests.post(config.URL_CLOUD_DNN_INFERENCE, json=data, timeout=config.timeout)
		response.raise_for_status()

	except Exception as e:
		print(e)
		return {"status": "error"}

	if (response.status_code != 200 and response.status_code != 201):
		return {"status": "error"}

	else:
		return {"status": "ok"}


def b_mobileNetInferenceEdge(tensor_img, p_tar):

	b_model.eval()
	with torch.no_grad():
		output, conf_list, infer_class = b_model.forwardExperiment(tensor_img.float(), p_tar, device)

	return output, conf_list, infer_class

def saveInferenceTime(inference_time, p_tar):
	
	result = {"inference_time": inference_time, "p_tar": p_tar}

	result_path = os.path.join(config.RESULTS_INFERENCE_TIME_EDGE, "inference_time.csv")

	if (not os.path.exists(result_path)):
		df = pd.DataFrame()
	else:
		df = pd.read_csv(result_path)	
		df = df.loc[:, ~df.columns.str.contains('^Unnamed')] 
	
	df = df.append(pd.Series(result), ignore_index=True)
	df.to_csv(result_path) 
