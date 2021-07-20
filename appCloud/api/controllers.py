from flask import Blueprint, g, render_template, request, jsonify, session, redirect, url_for, current_app as app
import json, os, time, sys, config, torch, requests
from .services import cloudProcessing


api = Blueprint("api", __name__, url_prefix="/api")


# Define url for the user send the image
@api.route('/cloud/cloudInference', methods=["POST"])
def cloud_inference():
	"""
	This function receives an image or feature map from edge device (Access Point)
	This functions is run in the cloud.
	"""

	data_from_edge = request.json
	result = cloudProcessing.dnnInferenceCloud(data_from_edge["feature"], data_from_edge["conf"], data_from_edge["p_tar"])

	if (result["status"] ==  "ok"):
		return jsonify(result), 200

	else:
		return jsonify(result), 500
