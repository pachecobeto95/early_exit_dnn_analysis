import requests, config


def send_json(url, json_data, wait_time=5):
	try:
		response = requests.post(url, json=json_data, timeout=wait_time)
		response.raise_for_status()

	except Exception as e:
		raise e

def main():
	url = "%s/api/edge/testJson"%(config.URL_EDGE)
	json_data = {"data": "teste"}
	send_json(url, json_data)



if __name__ == "__main__":
	main()