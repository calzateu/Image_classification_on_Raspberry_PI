import requests
import cv2
import time
import numpy as np
from time import sleep

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

# URL of the flask server
#url = "http://localhost:8000/classify"
url = "http://159.89.238.153:80/classify"

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()

    cv2.imshow("Image", img)
    k = cv2.waitKey(1)

    if k%256 == 27:
        break
    elif k%256 == 32:

        # encode image as jpeg
        _, img_encoded = cv2.imencode('.jpg', img)

        # prepare headers for http request
        content_type = 'image/jpeg'
        headers = {'content-type': content_type}

        # Send the request
        time1 = time.time()
        response = requests.post(url, data=img_encoded.tostring(), headers=headers)
        time2 = time.time()
        classification_time = np.round(time2-time1, 3)
        print("Classificaiton Time =", classification_time, "seconds.")

        # Print the response
        print(response)

        classification_label, prob = response.json()['label_id'], response.json()['prob']


         # Return the classification label of the image.
        text = "Image Label is: {0} with Accuracy: {1}%.".format(classification_label, prob)
        #print(text)

        show_result(img, text)
        cv2.waitKey(1)
        sleep(5)

        break


