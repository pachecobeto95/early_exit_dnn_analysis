import os

DIR_NAME = os.path.dirname(__file__)

DEBUG = True

# Edge URL Configuration 
HOST_EDGE = "192.168.0.20"
PORT_EDGE = 5000
URL_EDGE = "http://%s:%s"%(HOST_EDGE, PORT_EDGE)


# Cloud URL Configuration 
HOST_CLOUD = "192.168.0.20"
PORT_CLOUD = 3000
URL_CLOUD = "http://%s:%s"%(HOST_CLOUD, PORT_CLOUD)
URL_CLOUD_DNN_INFERENCE = "%s/api/cloud/cloudInference"%(URL_CLOUD)


#Dataset Path
dataset_path = os.path.join(DIR_NAME, "datasets", "256_ObjectCategories")

#Model Path
model_path = os.path.join(DIR_NAME, "appEdge", "api", "services", "models", "pristine_model_b_mobilenet_caltech.pth")


#width, height dimensions of the input image
input_shape = 300
resize_shape = 330


# timeout parameter
timeout = 5

#path of the interence time results
RESULTS_INFERENCE_TIME = os.path.join(DIR_NAME, "appEdge", "api", "services", "results")


# Settings to load B-Mobilenet model
n_classes = 258                   # number of classes in the dataset
exit_type = None                  # type of exit
pretrained = False                #always false
n_branches = 3                    #number of branches (early exits points)
