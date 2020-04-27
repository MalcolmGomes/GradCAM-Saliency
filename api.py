from flask import Flask, request, send_file
from flask_restful import Resource, Api
from saliency_mapper import *

import os
import requests
import torch
import time
import sys
from torchvision import models
from torchvision import transforms


app = Flask(__name__)
api = Api(app)

class SaliencyMapAPI(Resource):
	def get(self):
		filename = "malcolm.jpg"
		img = Image.open(filename)		
		output_path = generate_saliency_map(img, filename)
		return send_file(output_path, attachment_filename=filename)
	
	def post(self):
		image_url = request.form["image_url"]
		filename = image_url.split('/')[-1]
		r = requests.get(image_url, allow_redirects=True)
		open(filename, 'wb').write(r.content)           
		img = Image.open(filename)          
		output_path = generate_saliency_map(img, filename)
		os.remove(filename)
		return send_file(output_path, attachment_filename=filename)
		# return {
		# 	'img': output_path
		# }

api.add_resource(SaliencyMapAPI, '/')

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=80, debug=True)
