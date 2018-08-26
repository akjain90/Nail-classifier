# Nail-classifier
Nails classification challenge to classify good nails and bent nails based on convolution neural network

## Prerequisites

Python 3.5.5, Tenforflow 1.10, flask 1.0.2, OpenCV 3.4.1, sklearn 0.19.1, requests 2.19.1
For development: Anaconda or PyCharm

## Running
### Training
Put the training images in "../nailgun"
Create a directory "../model" to store the model and checkpoints during the training then go to the
directory "./training" and run the script "train.py":
```
python train.py
```
let the training finish. This will take some time.

### Running the app
Go to the directory "./app" and run "run_model_server.py"
```
python run_model_server.py
```
while the server is running open another terminal and run "request.py"
```
python request.py
```
once the execution is complete you will see the prediction as either "good" or "bent"
