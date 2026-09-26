from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load model
model = load_model("plant_disease_model.h5")

# Load data
datagen = ImageDataGenerator(rescale=1./255)

val_data = datagen.flow_from_directory(
    "dataset",
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical'
)

# Evaluate
loss, accuracy = model.evaluate(val_data)

print("Validation Accuracy:", accuracy)