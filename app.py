# =[Modules dan Packages]========================
from ultralytics import YOLO
from flask import request, Response, Flask, render_template, jsonify
from waitress import serve
from PIL import Image
import json
import numpy as np
from flask_wtf import FlaskForm
import secrets
import os
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename
from infer import *
from flask_ngrok import run_with_ngrok

# =[Variabel Global]=============================
app = Flask(__name__)
app.static_folder = 'static'

app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS']  = ['.jpg','.JPG', '.png', '.PNG', '.jpeg', '.JPEG']
app.config['UPLOAD_PATH']        = 'static/images/uploads/'

# NUM_CLASSES = 4
classes = ["crack", "horizontal crack", "vertical crack", "diagonal crack"]

# =[Routing]=====================================

# [Routing untuk Halaman Utama atau Home]
@app.route("/")
def beranda():
	return render_template('index.html')

# [Routing untuk API]	
@app.route("/api/deteksi",methods=['POST'])
def apiDeteksi():
    """
        Handler of /detect POST endpoint
        Receives uploaded file with a name "file", 
        passes it through YOLOv8 object detection 
        network and returns an array of bounding boxes.
        :return: a JSON array of objects bounding 
        boxes in format 
        [[x1,y1,x2,y2,object_type,probability],..]
    """
    buf = request.files["file"]
    filename  = secure_filename(buf.filename)
    
    if filename != '':
        file_ext        = os.path.splitext(filename)[1]
        gambar_prediksi = '/static/images/uploads/' + filename
        if file_ext in app.config['UPLOAD_EXTENSIONS']:
          buf.save(os.path.join(app.config['UPLOAD_PATH'], filename))
          boxes = detect_objects_on_image(Image.open(buf.stream))
          return Response(
            response=json.dumps({"gambar_prediksi": gambar_prediksi, "boxes": boxes}),
            mimetype='application/json'
          )
        else:
          return Response(
            response=json.dumps({"error":"File type not allowed"}),
            mimetype='application/json'
          )
    else:
      return Response(
        response=json.dumps({"error":"No file found"}),
        mimetype='application/json'
      )

def detect_objects_on_image(buf):
    """
    Function receives an image,
    passes it through YOLOv8 neural network
    and returns an array of detected objects
    and their bounding boxes
    :param buf: Input image file stream
    :return: Array of bounding boxes in format 
    [[x1,y1,x2,y2,object_type,probability],..]
    """
    model = YOLO("best.pt")
    results = model.predict(buf)
    result = results[0]
    output = []
    for box in result.boxes:
        x1, y1, x2, y2 = [
          round(x) for x in box.xyxy[0].tolist()
        ]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([
          x1, y1, x2, y2, result.names[class_id], prob
        ])
    return output

# =[Main]========================================		

if __name__ == '__main__':

	# # Run Flask di Google Colab menggunakan ngrok
	run_with_ngrok(app)
	app.run()
	


