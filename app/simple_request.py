# imports
import requests

# initialize the API endpoint URL along with the input image path
NAIL_TEST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "nail.jpeg"
# load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}
# submit the request
r = requests.post(NAIL_TEST_API_URL, files=payload).json()
print(r["prediction"])
