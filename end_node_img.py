import requests, config, os, json
from utils import LoadDataset

def send_img(url, json_data, imgPath):
	try:
		files = [
		('img', (imgPath, open(imgPath, 'rb'), 'application/octet')),
		('data', ('data', json.dumps(json_data), 'application/json')),]

		response = requests.post(url, files=files, timeout=config.timeout)
		response.raise_for_status()

	except Exception as e:
		raise e

def inferenceEdgeExp(url, json_data, datasetPath):
	dataset_dir_list = os.listdir(datasetPath)

	for j, dir_class in enumerate(dataset_dir_list):
		print(print("Number of Class: %s"%(j)))
		dir_path = os.path.join(datasetPath, dir_class)

		for i, img in enumerate(os.listdir(dir_path)):
			print("Image: %s"%(i))
			filePath = os.path.join(datasetPath, dir_class, img)
			send_img(url, json_data, filePath)

def main():
	url = "%s/api/edge/image_inference"%(config.URL_EDGE)
	p_tar = 0.8   #example to test
	json_data = {"p_tar": p_tar}
	input_dim = 300
	batch_size_test = 1
	normalization = False
	dataset_path = config.dataset_path
	#dataset = LoadDataset(input_dim, batch_size_test, normalization=normalization)
	#test_loader = dataset.caltech(config.dataset_path)
	inferenceEdgeExp(url, json_data, dataset_path)



if __name__ == "__main__":
	main()
