from keras.preprocessing import image
import numpy as np

import keras


img=image.load_img("image.png",target_size=(48,48))

img=image.img_to_array(img)

img=img[:,:,1]

img=img/255.0



emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

model = keras.models.load_model("saved_model")

import matplotlib.pyplot as plt

plt.imshow(img,cmap="gray")
plt.show()


pr=model.predict_classes([img.reshape(1,48,48,1)])

print(pr)
print(emotion_dict.get(pr[0]))

