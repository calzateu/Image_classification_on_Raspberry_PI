import cv2

import tensorflow as tf
from PIL import Image
import numpy as np
import time
from time import sleep


def load_labels(path): # Read the labels from the text file as a Python list.
    with open(path, 'r') as f:
        return [line.strip() for i, line in enumerate(f.readlines())]

def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path)
    #print("Model Loaded Successfully.")

    interpreter.allocate_tensors()

    return interpreter

def preprocess(img):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)
    return input_data

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
    set_input_tensor(interpreter, image)

    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    #scale, zero_point = output_details['quantization']
    #print(zero_point)
    #output = scale * (output - zero_point)
    #print(scale)

    ordered = np.argpartition(-output, 1)
    return [(i, output[i]) for i in ordered[:top_k]][0]

def show_result(img, text):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = (50, 50)

    # fontSc
    fontScale = 0.63

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.putText() method
    cv2.putText(img, text, org, font,
                    fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow("Image", img)




cap = cv2.VideoCapture(0)

data_folder = "Segunda_etapa/Modelos/VGG16"
model_path = data_folder + "/model_VGG16_quant_f16.tflite"
label_path = data_folder + "/labels.txt"

# Read class labels.
labels = load_labels(label_path)
interpreter = load_model(model_path)

_, height, width, _ = interpreter.get_input_details()[0]['shape']
print("Image Shape (", width, ",", height, ")")

while True:
    _, img = cap.read()
    #img = cv2.imread("Image.jpg")

    cv2.imshow("Image", img)
    k = cv2.waitKey(1)

    if k%256 == 27:
        break
    elif k%256 == 32:
        input_data = preprocess(img)

        # Classify the image.
        time1 = time.time()
        label_id, prob = classify_image(interpreter, input_data)#img)
        time2 = time.time()
        classification_time = np.round(time2-time1, 3)
        print("Classificaiton Time =", classification_time, "seconds.")


        # Return the classification label of the image.
        classification_label = labels[label_id]
        text = "Image Label is: {0} with Accuracy: {1}%.".format(classification_label, np.round(prob*100, 2))
        print(text)

        show_result(img, text)
        k = cv2.waitKey(1)
        sleep(5)

        break

