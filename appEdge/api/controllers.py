from flask import Blueprint, g, render_template, request, jsonify, session, redirect, url_for, current_app as app
import json, os, time, sys, config, torch, requests
from .services import edgeProcessing

api = Blueprint("api", __name__, url_prefix="/api")

# Define url for the user send the image
@api.route('/edge/image_inference', methods=["POST"])
def edge_receive_img_robust():
	"""
	This function receives an image from user or client with smartphone at the edge device into smart sity context
	"""	
	fileImg = request.files['img']

	json_data = json.load(request.files['data'])
	p_tar = json_data["p_tar"]

	#This functions process the DNN inference
	result = edgeProcessing.edgeInference(fileImg, p_tar)

	if (result["status"] ==  "ok"):
		return jsonify(result), 200

	else:
		return jsonify(result), 500


@api.route('/edge/testJson', methods=["POST"])
def edge_test_json():
	"""
	This function tests the server to receive a simple json post request.
	"""	

	post_data = request.json

	result = {"status": "ok"}

	if (result["status"] ==  "ok"):
		return jsonify(result), 200

	else:
		return jsonify(result), 500
