import cv2
from keras.models import load_model
import numpy as np
import keras
import matplotlib.pyplot as plt



model=load_model("saved_model")



img=keras.preprocessing.image.load_img("image.png",target_size=(48,48))

emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}


img=keras.preprocessing.image.img_to_array(img)


print(img.shape)

predict=model.predict_classes([img[:,:,1].reshape(1,48,48,1)])
print(predict)

plt.imshow(img[:,:,2])
plt.show()


