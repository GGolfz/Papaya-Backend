from flask import Flask,request
from flask_cors import CORS
import cv2
import io
import base64
import numpy as np
from PIL import Image
from keras.models import load_model
import tensorflow as tf
import json
app = Flask(__name__)
CORS(app)
model_data = [['model_cnn.h5',150],['model_vgg.h5',224]]
@app.route('/papaya-api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        img = np.array(Image.open(io.BytesIO(base64.b64decode(data['data'].split('base64,')[1]))).convert('RGB'))
        data = detect_papaya(img)
        prediction_data = []
        prediction = prediction_img(img)
        prediction_data.append({'type':'original','class':prediction[0],'confident':float(prediction[1])})
        try:
            for i in data:
                print(i)
                x_start = i['x']
                y_start = i['y']   
                x_end = round(x_start + i['w'])
                y_end = round(y_start + i['h'])
                crop_img = img[y_start:y_end+1, x_start:x_end+1]
                bgr = crop_img[...,::-1].copy()
                prediction = prediction_img(crop_img)
                string = 'data:image/png;base64,'+base64.b64encode(cv2.imencode('.jpg',bgr)[1]).decode()
                prediction_data.append({'type':'cropped','img':string,'class':prediction[0],'confident':float(prediction[1])})
        except:
            print("Error when detection")
    except:
        return {"error":True}
    return json.dumps(prediction_data)
    

def prediction_img(image):
    class_labels = ["ripe", "medium","not ripe"]
    scores = []
    for data in models:
        model = data[0]
        IMAGE_SIZE = data[1]
        img = cv2.resize(image,(IMAGE_SIZE,IMAGE_SIZE))
        img = img / 255
        img = img.reshape(1,IMAGE_SIZE,IMAGE_SIZE,3)
        predictions = model.predict(img)
        scores.append(predictions[0])
    scores = np.array(scores)
    print(scores)
    scores = np.mean(scores,axis=0)
    print(scores)
    pred_class = class_labels[np.argmax(scores)]
    return [pred_class,np.max(scores)]

def detect_papaya(img):
    classes = ['papaya']

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    r = blob[0, 0, :, :]

    net.setInput(blob)
    outputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []
    h, w = img.shape[:2]

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.55:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    data = []
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            if x >= 0 and y>=0 and w>=0 and h>=0:
                data.append({'x':x,'y':y,'w':w,'h':h,'confident':confidences[i]})
                if len(data) == 3:
                    break
    return data

if __name__ == '__main__':
    models = [[load_model(config[0]),config[1]] for config in model_data]
    net = cv2.dnn.readNetFromDarknet('custom-yolov4-detector.cfg','custom-yolov4-detector_best.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    app.run(debug=False, host='0.0.0.0',port=5002)