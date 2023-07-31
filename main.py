from flask import Flask, render_template, request
from keras.models import load_model
import cv2
import numpy as np

app = Flask(__name__)

dic = {0 : 'Bakso', 1 : 'Bebek Betutu', 2 : 'Gado gado', 3 : 'Gudeg', 4 : 'Nasi Goreng'
       , 5 : 'Pempek', 6:'Rawon',7:'Rendang',8:'Sate',9:'Soto'}

model = load_model('EfficientNet_Adam_20')

def predict_label(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (380,380), interpolation = cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0).astype(np.float32)
    pred = model.predict(img)
    prediction  = np.argmax(pred, axis=1)
    return dic[int(prediction)]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("home.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		if(img_path=="static/"):
			return render_template("home.html", no_pic = "No Pic")
		
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("home.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)