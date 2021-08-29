import os

DIR_NAME = os.path.dirname(__file__)

DEBUG = True

# Edge URL Configuration 
HOST_EDGE = "192.168.0.20"
PORT_EDGE = 5000
URL_EDGE = "http://%s:%s"%(HOST_EDGE, PORT_EDGE)
URL_EDGE_DNN_INFERENCE = "%s/api/edge/edgeInference"%(URL_EDGE)
urlConfModelEdge = "%s/api/edge/modelConfiguration"%(URL_EDGE)


# Cloud URL Configuration 
HOST_CLOUD = "192.168.0.20"
PORT_CLOUD = 3000
URL_CLOUD = "http://%s:%s"%(HOST_CLOUD, PORT_CLOUD)
URL_CLOUD_DNN_INFERENCE = "%s/api/cloud/cloudInference"%(URL_CLOUD)
urlConfModelCloud = "%s/api/cloud/modelConfiguration"%(URL_CLOUD)


#Dataset Path
dataset_path = os.path.join(DIR_NAME, "datasets", "256_ObjectCategories")

#Model Path
model_path = os.path.join(DIR_NAME, "appEdge", "api", "services", "models", "pristine_model_b_mobilenet_caltech.pth")
edge_model_path = os.path.join(DIR_NAME, "appEdge", "api", "services", "models")
cloud_model_path = os.path.join(DIR_NAME, "appCloud", "api", "services", "models")


#width, height dimensions of the input image
resize_shape = 330


# timeout parameter
timeout = 5

#path of the interence time results
RESULTS_INFERENCE_TIME_EDGE = os.path.join(DIR_NAME, "appEdge", "api", "services", "results")
temp_edge_path = os.path.join(DIR_NAME, "appEdge", "api", "services", "temperature")
temp_cloud_path = os.path.join(DIR_NAME, "appCloud", "api", "services", "temperature")

# Settings to load B-Mobilenet model
n_classes = 258                   # number of classes in the dataset
exit_type = ""                  # type of exit
pretrained = False                #always false
distribution = "linear"
exit_type = "bnpool"
input_dim = 300
input_shape = (3, input_dim, input_dim)
nr_max_branches = 5
nr_min_branches = 2
model_name = "mobilenet"
dataset_name = "caltech256"
nr_branch_model = 5

model_path_edge_final = os.path.join(DIR_NAME, "appEdge", "api", "services", "models", 
	"mobilenet_%s_2_branches_%s.pth"%(dataset_name, nr_branch_model))

