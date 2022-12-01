import cv2

import tensorflow as tf
from PIL import Image
import numpy as np
import time

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()


    def load_labels(path): # Read the labels from the text file as a Python list.
        with open(path, 'r') as f:
            return [line.strip() for i, line in enumerate(f.readlines())]

    def set_input_tensor(interpreter, image):
        tensor_index = interpreter.get_input_details()[0]['index']
        input_tensor = interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def classify_image(interpreter, image, top_k=1):
        set_input_tensor(interpreter, image)

        interpreter.invoke()
        output_details = interpreter.get_output_details()[0]
        output = np.squeeze(interpreter.get_tensor(output_details['index']))

        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)

        ordered = np.argpartition(-output, 1)
        return [(i, output[i]) for i in ordered[:top_k]][0]

    data_folder = "/home/cristian/Descargas/TRABAJO/Trabajos_ciencia_datos/Clasificacion_imagenes/Segunda_etapa/Modelos/MobileNet"

    model_path = data_folder + "/model_MobileNet.tflite"
    label_path = data_folder + "/labels.txt"

    interpreter = tf.lite.Interpreter(model_path)
    print("Model Loaded Successfully.")

    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    print("Image Shape (", width, ",", height, ")")

    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imH, imW, _ = img.shape
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # Load an image to be classified.
    #image = Image.open(data_folder + "test.jpg").convert('RGB').resize((width, height))

    # Classify the image.
    time1 = time.time()
    label_id, prob = classify_image(interpreter, input_data)#img)
    time2 = time.time()
    classification_time = np.round(time2-time1, 3)
    print("Classificaiton Time =", classification_time, "seconds.")

    # Read class labels.
    labels = load_labels(label_path)

    # Return the classification label of the image.
    classification_label = labels[label_id]
    text = "Image Label is: {0} with Accuracy: {1}%.".format(classification_label, np.round(prob*100, 2))
    print(text)

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
    image = cv2.putText(img, text, org, font,
                    fontScale, color, thickness, cv2.LINE_AA)


    cv2.imshow("Image", img)
    cv2.waitKey(1)
