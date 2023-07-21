from fastapi import FastAPI, File, UploadFile, HTTPException
from torchvision import transforms
import onnxruntime
import numpy as np
import io
from PIL import Image
import uvicorn
import syslog
import sys
import requests
from torchvision.transforms.functional import InterpolationMode

# https://docs.python.org/3/library/syslog.html
syslog.openlog(ident="Writingtype-API", logoption=syslog.LOG_PID, facility=syslog.LOG_LOCAL0)

# Path to pretrained model
MODEL_PATH = './model/writing_type_v1.onnx'
# Input image size
IMG_SIZE = 224

# Transformations used for input images
img_transforms = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.882, 0.883, 0.899], [0.088, 0.089, 0.094])
    ])

# Predicted class labels
classes = {0:"Handwritten", 1:"Typewritten", 2:"Combination"}

try:
    # Initialize API Server
    app = FastAPI()
except Exception as e:
    syslog.syslog(syslog.LOG_ERR, 'Failed to start the API server: {}'.format(e))
    sys.exit(1)


# Function is run (only) before the application starts
@app.on_event("startup")
async def load_model():
    """
    Load the pretrained model on startup.
    """
    try:
        # Load the onnx model and the trained weights
        model = onnxruntime.InferenceSession(MODEL_PATH)
        # Add model to app state
        app.package = {"model": model}
    except Exception as e:
        syslog.syslog(syslog.LOG_ERR, 'Failed to load the model file: {}'.format(e))
        raise HTTPException(status_code=500, detail=f"Failed to load the model file: {e}")

def softmax(x):
    return(np.exp(x)/np.exp(x).sum())

def predict(image):
    """
    Perform prediction on input image.
    """
    # Get model from app state
    model = app.package["model"]
    image = img_transforms(image.convert("RGB")).unsqueeze(0)
    # Transform tensor to numpy array
    img = image.detach().cpu().numpy()
    input = {model.get_inputs()[0].name: img}
    # Run model prediction
    output = model.run(None, input)
    # Get predicted class
    pred = np.argmax(output[0], 1)
    pred_class = pred.item()
    # Get the confidence value for the prediction
    pred_confidences = softmax(output[0][0])[pred_class]
    # Confidence of the prediction as %
    pred_confidences = float(pred_confidences)
    # Return predicted class and confidence in dictionary form
    predictions = {'prediction': classes[pred_class], 'confidence': pred_confidences}

    return predictions


# Endpoint for POST requests: input image is received with the http request
@app.post("/writingtype")
async def postit(file: UploadFile = File(...)):
    try:
        # Loads the image sent with the POST request
        req_content = await file.read()
        image = Image.open(io.BytesIO(req_content))
        image.draft('RGB', (IMG_SIZE, IMG_SIZE))
    except Exception as e:
        syslog.syslog(syslog.LOG_ERR, 'Failed to load the input image file: {}'.format(e)) 
        raise HTTPException(status_code=400, detail=f"Failed to load the input image file: {e}")

    # Get predicted class and confidence
    try: 
        predictions = predict(image)
    except Exception as e:
        syslog.syslog(syslog.LOG_ERR, 'Failed to analyze the input image file: {}'.format(e))
        raise HTTPException(status_code=500, detail=f"Failed to analyze the input image file: {e}")
        
    return predictions

# Endpoint for GET requests: input image path is received with the http request
@app.get("/writingtype_path")
async def postit_url(path: str):
    try:
        # Loads the image from the path sent with the GET request
        #req_content = requests.get(url)
        image = Image.open(path)
        image.draft('RGB', (IMG_SIZE, IMG_SIZE))

    except Exception as e:
        syslog.syslog(syslog.LOG_ERR, 'Failed to recognize file {} as a path. Error: {}'.format(path, e))
        raise HTTPException(status_code=400, detail=f"Failed to load the input image file: {e}")

    # Get predicted class and confidence
    try: 
        predictions = predict(image)
    except Exception as e:
        syslog.syslog(syslog.LOG_ERR, 'Failed to analyze the input image file: {}'.format(e))
        raise HTTPException(status_code=500, detail=f"Failed to analyze the input image file: {e}")

    return predictions

# Endpoint for GET requests: input image url is received with the http request
@app.get("/writingtype_url")
async def postit_url(url: str):
    try:
        # Loads the image from the path sent with the GET request
        req_content = requests.get(url)
        image = Image.open(io.BytesIO(req_content.content))
        image.draft('RGB', (IMG_SIZE, IMG_SIZE))

    except Exception as e:
        syslog.syslog(syslog.LOG_ERR, 'Failed to recognize file {} as an url. Error: {}'.format(url, e))
        raise HTTPException(status_code=400, detail=f"Failed to load the input image file: {e}")

    # Get predicted class and confidence
    try: 
        predictions = predict(image)
    except Exception as e:
        syslog.syslog(syslog.LOG_ERR, 'Failed to analyze the input image file: {}'.format(e))
        raise HTTPException(status_code=500, detail=f"Failed to analyze the input image file: {e}")

    return predictions

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level="info")
