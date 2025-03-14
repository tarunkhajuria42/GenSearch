import tensorflow as tf
import numpy as np
class fmnist_classifier:
    def __init__(self):
        self.model = tf.keras.models.load_model("f_mnist")
        return
    def predict(self,img):
        if(len(img.shape)==2):
            img = np.expand_dims(img,axis =0)
        cls = np.argmax(self.model.predict(img))
        return cls
        