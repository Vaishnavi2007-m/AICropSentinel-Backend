import cv2
from gradcam import get_gradcam
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = load_model("plant_disease_model.h5")
# 🔥 FORCE BUILD MODEL
dummy_input = np.zeros((1,128,128,3))
model(dummy_input)

# Class names (VERY IMPORTANT - match dataset folder order)
class_names = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites",
    "Tomato_Target_Spot",
    "Tomato_Yellow_Leaf_Curl",
    "Tomato_Mosaic_virus",
    "Tomato_healthy"
]

# Load image
img_path = "test6.JPG"

img = image.load_img(img_path, target_size=(128,128))
img_array = image.img_to_array(img)/255
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

predicted_index = np.argmax(prediction)
confidence = np.max(prediction) * 100


print("Disease:", class_names[predicted_index])
print("Accuracy: {:.2f}%".format(confidence))

# 🔥 Grad-CAM

last_conv_layer_name = "conv2d_2"   # change if different

heatmap = get_gradcam(img_array, model, last_conv_layer_name)

# Resize heatmap
heatmap = cv2.resize(heatmap, (128,128))
heatmap = np.uint8(255 * heatmap)

# Apply color map
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Load original image
original = cv2.imread("test6.jpg")
original = cv2.resize(original, (128,128))

# Overlay heatmap
overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

# Show result
cv2.imshow("Explainable AI - Affected Area", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()