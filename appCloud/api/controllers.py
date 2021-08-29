from flask import Blueprint, g, render_template, request, jsonify, session, redirect, url_for, current_app as app
import json, os, time, sys, config, torch, requests
from .services import cloudProcessing
from .services.cloudProcessing import model


api = Blueprint("api", __name__, url_prefix="/api")



@api.route("/cloud/modelConfiguration", methods=["POST"])
def cloudModelConfiguration():
	data = request.json
	model.nr_branches_model = data["nr_branches_model"]
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.update_load_model(device, data["dataset_name"])
	model.load_temperature()
	return jsonify({"status": "ok"}), 200



# Define url for the user send the image
@api.route('/cloud/cloudInference', methods=["POST"])
def cloud_inference():
	"""
	This function receives an image or feature map from edge device (Access Point)
	This functions is run in the cloud.
	"""

	data_from_edge = request.json
	result = cloudProcessing.dnnInferenceCloud(data_from_edge["feature"], data_from_edge["conf"], data_from_edge["class_list"], 
		data_from_edge["p_tar"], data_from_edge["nr_branch_edge"])

	if (result["status"] ==  "ok"):
		return jsonify(result), 200

	else:
		return jsonify(result), 500
