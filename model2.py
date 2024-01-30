import pickle
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import random

# Load the model using pickle
with open("D:/INTERNSHIP/lung_health_model_architecture.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load and preprocess the image
img = image.load_img("C:/Users/LENOVO/Downloads/archive(2)/chest_xray/val/PNEUMONIA/person1946_bacteria_4875.jpeg", target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
img_data = preprocess_input(x)

# Make predictions
classes = model.predict(img_data)
print(classes)

if classes[0] == 1:
    print("It is not normal, need to take a test via a doctor")
    doctors = ["We recommend consulting with Dr. Yash for further evaluation.",
               "Considering the diagnosis of Pneumonia, Dr. Yash may provide specific advice.",
               "Dr. Williams"]
    print(random.choice(doctors))
else:
    print("It is normal")
