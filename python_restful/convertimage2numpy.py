import cv2
import numpy as np
def getImage(name):
    #fetch image
    img = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
    if img is None:
            return None
    #standard input is batchsize x 3 x 224 x 224
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    return img
def getImageFromFile(fr):
    # fr = f.read()
    image = np.asarray(bytearray(fr), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if img is None:
            return None
    #standard input is batchsize x 3 x 224 x 224
    img = cv2.resize(img, (224, 224))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    return img

np.save(open("upload/data","wb"),getImage("upload/testing_machine.JPG"))