import os, config, time
import requests, sys, json, os
import numpy as np
from PIL import Image
import pandas as pd
import argparse
from utils import LoadDataset
from requests.exceptions import HTTPError, ConnectTimeout
from glob import glob
import torch

def sendModelConf(url, nr_branches_model, dataset_name):
	

	data_dict = {"nr_branches_model": nr_branches_model, "dataset_name": dataset_name}

	try:
		r = requests.post(url, json=data_dict, timeout=30)
		r.raise_for_status()
	
	except HTTPError as http_err:
		raise SystemExit(http_err)

	except ConnectTimeout as timeout_err:
		print("Timeout error: ", timeout_err)

def sendImage(img_path, url, p_tar, nr_branch_edge):
	data_dict = {"p_tar": p_tar, "nr_branch_edge": nr_branch_edge}

	files = [('img', (img_path, open(img_path, 'rb'), 'application/octet')),
	('data', ('data', json.dumps(data_dict), 'application/json')),]

	try:
		r = requests.post(url, files=files, timeout=config.timeout)
		r.raise_for_status()
	
	except HTTPError as http_err:
		raise SystemExit(http_err)

	except ConnectTimeout as timeout_err:
		print("Url: ¨%s, Timeout error: %s"%(url, timeout_err))

def inferenceTimeExperiment(imgs_files_list, url_edge, p_tar_list, nr_branch_edge_list):
	for (i, img_path) in enumerate(imgs_files_list, 1):
		print("Img: %s"%(i))

		# This loop varies the number of branches processed at the cloud
		for nr_branch_edge in nr_branch_edge_list:

			# For a given number of branches processed in edge, this loop changes the threshold p_tar configuration.
			for p_tar in p_tar_list:
				sendImage(img_path, url_edge, p_tar, float(nr_branch_edge))
				#sys.exit()

 

def main(args):
	#Number of side branches that exists in the early-exit DNNs
	#nr_branches_model_list = np.arange(config.nr_min_branches, config.nr_max_branches+1)

	nr_branches_model = args.n_branches

	imgs_files_list = list(glob(os.path.join(config.dataset_path, "*", "*")))

	p_tar_list = [0.7, 0.8, 0.85, 0.9]

	#for nr_branches_model in nr_branches_model_list:

	#This line defines the number of side branches processed at the cloud
	nr_branch_edge = np.arange(2, nr_branches_model+1)
	sendModelConf(config.urlConfModelEdge, nr_branches_model, args.dataset_name)
	sendModelConf(config.urlConfModelCloud, nr_branches_model, args.dataset_name)
	inferenceTimeExperiment(imgs_files_list, config.URL_EDGE_DNN_INFERENCE, p_tar_list, nr_branch_edge)


if (__name__ == "__main__"):
	# Input Arguments. Hyperparameters configuration
	parser = argparse.ArgumentParser(description="Evaluating early-exits DNNs perfomance")
	parser.add_argument('--n_branches', type=int, default=config.nr_max_branches,
		choices=list(range(config.nr_min_branches, config.nr_max_branches+1)), 
		help='Number of branches in the early-exit DNNs model (default: %s)'%(config.nr_max_branches))

	parser.add_argument('--dataset_name', type=str, default="caltech256",choices=["caltech256", "cifar10", "cifar100"], 
		help='Dataset name (default: Caltech-256)')

	args = parser.parse_args()

	main(args)
