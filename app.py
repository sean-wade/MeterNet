# -*- coding:utf-8 -*-
import os
import time
import base64
import random

import cv2
import numpy as np
from flask import Flask, g
from flask import request, Response, render_template

from MeterNet.features import Extractor
from database.faceSQLite import FaceSQL
from MeterNet.distance import NumpyDistance
from utils import string_util, file_util


app = Flask(__name__, template_folder="./templates")

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

Margin = 1.99

extractor = Extractor(checkpoint='./weights/checkpoint_49.pth')
#extractor = Extractor(center_type = False, embedding_size=256, checkpoint='./weights/checkpoint_95.pth')    #Not good

faceDB = FaceSQL("Meter.db")
all_rows = faceDB.queryAll()
meter_names = [all_rows[i][1] for i in range(len(all_rows))]
print(meter_names)

l2_dist = NumpyDistance(2)

@app.route("/")
def start_page():
    print("Start")
    return render_template('detect.html')


def decode_save_img(receive_file, save_path):
    image = cv2.imdecode(np.fromstring(receive_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    cv2.imwrite(save_path, image)

    image_content = cv2.imencode('.jpg', image)[1].tostring()
    encoded_image = base64.encodestring(image_content)
    to_send = 'data:image/jpg;base64, ' + str(encoded_image, 'utf-8')
    return to_send


@app.route("/delete_row", methods=['POST'])
def delete_row():
    global all_rows
    global meter_names
    try:
        print(request.form)
        name = request.form.get('name')
        result = ""
        if string_util.is_empty(name):
            result = "Name Empty! ====>>>> "
            raise Exception("Name Empty!")

        if name not in meter_names:
            result = "Name Not in DB! ====>>>> "
            raise Exception("Name Not in DB!")

        
        all_rows = faceDB.deleteName(name)
        meter_names = [all_rows[i][1] for i in range(len(all_rows))]
        print(meter_names)

        Delete_status = True
        
        result = "Success delete: %s ====>>>> " % name

    except Exception as e:
        Delete_status = False
        if result=="":
            result = "Unknown Wrong! Please Check Code!!! ====>>>> "
        print(e)
    
    result += ' - '.join(meter_names)

    return render_template('detect.html', meterDetected=Delete_status, meter_name=result, image_to_show=None, init=True)


@app.route("/do_sign_up", methods=['POST'])
def do_sign_up():
    global all_rows
    global meter_names
    try:
        print(request.form)
        name = request.form.get('name')
        file = request.files['image']
        if not file:
            result = "File Empty! ====>>>> "
            raise Exception("File Empty!")

        if string_util.is_empty(name):
            result = "Name Empty! ====>>>> "
            raise Exception("Name Empty!")

        if name in meter_names:
            result = "Name Already in DB! ====>>>> "
            raise Exception("Name Already in DB!")

        save_path = "register/" + file.filename
        to_send = decode_save_img(file, save_path)
        

        feature = extractor.get_feature(save_path)
        identifyID = Extractor.vector_to_str(feature)
        faceDB.insert(len(all_rows), name, identifyID)
        all_rows = faceDB.queryAll()
        meter_names = [all_rows[i][1] for i in range(len(all_rows))]
        print(meter_names)
        result = "Register Success! ====>>>> "
        result += ' - '.join(meter_names)
        Register_status = True

    except Exception as e:
        #result = "Register Failed!"
        Register_status = False
        to_send = None
        print(e)
    return render_template('detect.html', meterDetected=Register_status, meter_name=result, image_to_show=to_send, init=True)


@app.route('/detect', methods=['POST'])
def upload_file():
    global extractor
    try:
        file = request.files['image']
        result = ""
        if not file:
            result = "File Empty!"
            raise Exception("File Empty!")

        save_path = "receive/" + file.filename
        to_send = decode_save_img(file, save_path)

        feature = extractor.get_feature(save_path)

        distances = []
        for db_row in all_rows:
            db_feature = Extractor.str_to_vector(db_row[2])
            dist = l2_dist.forward(feature, db_feature)
            distances.append(dist)
        min_idx = np.argmin(np.array(distances))
        print("Min Dist Meter: ", all_rows[min_idx][1], distances[min_idx])

        is_Meter = (0 < distances[min_idx] < Margin)
        result = all_rows[min_idx][1]    # if is_Meter else "unknown"
        result = result + "     Meter Type !!! Min Distance: " + str(distances[min_idx])

    except Exception as e:
        is_Meter = False
        if result=="":
            result = "Unknown Wrong! Please Check Code!!!"
        to_send = None
        print(e)
    return render_template('detect.html', meterDetected=is_Meter, meter_name=result, image_to_show=to_send, init=True)


app.run(host='192.168.197.146',
        port=7890,
        debug=False,)
        #threaded=True)



