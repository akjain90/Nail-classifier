# imports
import numpy as np
import cv2
import tensorflow as tf
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from PIL import Image
import numpy as np
import flask
import io

# initialize our Flask application and the tensorflow model
app = flask.Flask(__name__)
graph = None
saver = None
sess = None


def load_model(final_model_path):
    # load the pre-trained model
    global graph
    global saver
    global sess
    graph = tf.get_default_graph()
    saver = tf.train.import_meta_graph(final_model_path + '.meta')
    saver.restore(sess, final_model_path)


def prepare_image(image):
    # Prepare the image before feeding to network for prediction
	temp = np.array(image)
	temp = temp[235:985, 600:1350]
	ret, thresh = cv2.threshold(temp, 127, 255, cv2.THRESH_TRUNC)
	thresh = (thresh - np.mean(thresh)) / np.std(thresh)
	temp_a = cv2.resize(cv2.GaussianBlur(thresh, (5, 5), 1), (500, 500))
	temp_b = cv2.resize(cv2.GaussianBlur(temp_a, (5, 5), 1), (250, 250))
	return temp_b.reshape(-1, 250 * 250)


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned
	data = {"success": False}
	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))
			# preprocess the image and prepare it for classification
			image = prepare_image(image)

			# classify the input image and then initialize
			prediction = graph.get_tensor_by_name(name="predict:0")
			sample_predict = sess.run(prediction, feed_dict={"X:0": image})
			label_dict = {"0": "bent", "1": "good"}

			data = {"prediction": label_dict[str(sample_predict[0])]}
			# indicate that the request was a success
			data["success"] = True
	# return the data dictionary as a JSON response
	return flask.jsonify(data)

if __name__ == "__main__":
    proj_dir = "../../model/"
    final_model_path = proj_dir + "my_model"
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("Restoring")
        load_model(final_model_path)
        app.run(host="localhost", port=5000)
