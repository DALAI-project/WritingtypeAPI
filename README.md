# WritingtypeAPI

<img src="https://github.com/DALAI-hanke/WritingtypeAPI/assets/33789802/c855c031-9fd4-44ba-bd73-980b15bbd224.jpg"  width="60%" height="60%">

API for a machine learning model trained to detect writingtype from scanned document images. 
The user sends the API an input image (in .jpg, .png or .tiff format) of a scanned document, and the API returns a reply 
containing the predicted classification ('handwritten', 'typewritten' or 'combination'), and the corresponding prediction confidence (a number
between 0 and 1).   

## Model training and testing 

The neural network model used for the image classification task was built using the PyTorch library, and the model training
was done by fine-tuning an existing [Densenet neural network model](https://pytorch.org/vision/main/models/generated/torchvision.models.densenet121.html).
The trained model file was transformed into the [ONNX](https://onnx.ai/) format in order to speed up inference and to make the use of the model less dependent on specific frameworks and libraries. 

Class|Training samples|Validation samples|Test samples|Test accuracy
-|-|-|-|-
Handwritten|19 816|2 202|3 889|97.3%
Typewritten|13 135|1 460|2 574|98.5%
Combination|16 983|1 887|3 329|96.2%

The model has been trained and tested using 49 934 scanned document images. With a test set of over 9 792 images, the model reaches over 97% detection accuracy for all classes. 

The data used in model training and testing contain sensitive data, and therefore the dataset can not be made publicly available.

## Running the API

The API code has been built using the [FastAPI](https://fastapi.tiangolo.com/) library. It can be run either in a virtual environment,
or in a Docker container. Instructions for both options are given below. 

The API uses the pretrained machine learning model file located in the `/model` folder. By default the file name should be `writing_type_v1.onnx`.
If you use a model with different name, you need to update the model name in the `MODEL_PATH` variable of the `api.py` file.

### Running the API in a virtual environment

These instructions use a conda virtual environment, and as a precondition you should have Miniconda or Anaconda installed on your operating system. 
More information on the installation is available [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). 

#### Create and activate conda environment using the following commands:

`conda create -n writingtype_api_env python=3.7`

`conda activate writingtype_api_env`

#### Install dependencies listed in the *requirements.txt* file:

`pip install -r requirements.txt`

#### Start the API running a single process (with Uvicorn server):

Using default host: 0.0.0.0, default port: 8000

`uvicorn api:app`
 
Select different host / port:

`uvicorn api:app --host 0.0.0.0 --port 8080`

#### You can also start the API with Gunicorn as the process manager (find more information [here](https://fastapi.tiangolo.com/deployment/server-workers/)) (NB! does not work on Windows):

`gunicorn api:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8080`

  - workers: The number of worker processes to use, each will run a Uvicorn worker

  - worker-class: The Gunicorn-compatible worker class to use in the worker processes

  - bind: This tells Gunicorn the IP and the port to listen to, using a colon (:) to separate the IP and the port

### Running the API using Docker

As a precondition, you should have Docker Engine installed. More information on the installation can be found [here](https://docs.docker.com/engine/install/). 

#### Build Docker image using the *Dockerfile* included in the repository: 

`docker build -t writingtype_image .`

Here the new image is named writingtype_image. After successfully creating the image, you can find it in the list of images by typing `docker image ls`.

#### Create and run a container based on the image:

`sudo docker run -d --name writingtype_container -p 8000:8000 writingtype_image`

In the Dockerfile, port 8000 is exposed, meaning that the container listens to that port. In the above command, the corresponding host port can be chosen as the first element in `-p <host-port>:<container-port>`. If only the container port is specified, Docker will automatically select a free port as the host port. 
The port mapping of the container can be viewed with the command `sudo docker port writingtype_container`

## Logging

Logging events are saved into a file `api_log.log` in the same folder where the `api.py` file is located. Previous content of the log file is overwritten after each restart. More information on different logging options is available [here](https://docs.python.org/3/library/logging.html).

## Testing the API

The API has three endpoints: `/writingtype` endpoint expects the input image to be included in the client's POST request, while  
`/writingtype_path` endpoint expects to receive the filepath to the image as a query parameter, and lastly `/writingtype_url` expects to receive the url to the image as a query parameter.

### Testing the API in a virtual environment

You can test the `/writingtype` endpoint of the API for example using curl:

`curl http://127.0.0.1:8000/writingtype -F file=@/path/img.jpg`

NB! Windows users might encounter following error `Invoke-WebRequest : A parameter cannot be found that matches parameter name 'F'.`. This can be bypassed by running a command `Remove-item alias:curl`.

The second option is to send the path to the image file with the http request:

`curl http://127.0.0.1:8000/writingtype_path?path=/path/img.jpg`

The third option is to send the url to the image file with the http request:

`curl http://127.0.0.1:8000/writingtype_url?url=/url/to/img.jpg`

The host and port should be the same ones that were defined when starting the API.
The image path `/path/img.jpg` should be replaced with a path to the image that is used as test input. 

### Testing the API using Docker

In the Docker version of the API, it is easiest to use the `/writingtype` endpoint of the API. This can be tested 
for example using curl:

`curl http://127.0.0.1:8000/writingtype -F file=@/path/img.jpg`

Sending the url/path to the image file with the http request to the API requires 
the use of [bind mount](https://docs.docker.com/storage/bind-mounts/) to mount the desired file or 
directory into the Docker container. For instance if the input images are located in a local folder 
`/home/user/data`, the container can be created and started the using the command 

`docker run -v /home/user/data:/data -d --name writingtype_container -p 8000:8000 writingtype_image`

and then the image paths can be sent to the API with the http request:

`curl http://127.0.0.1:8000/writingtype_path?path=/data/img.jpg`

### Output of the API

The output is in a .json form and consists of the predicted class label and the confidence for the prediction.
So for instance the output could be 

`{"prediction":"Typewritten","confidence":0.995205283164978}`



