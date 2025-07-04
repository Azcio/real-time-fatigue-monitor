import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

trainPath = "Drowsy_datset/train"
testPath= "Drowsy_datset/test"

TrainData = ImageDataGenerator(rescale=1./255, 
                                   rotation_range=20, 
                                   zoom_range=0.2, 
                                   horizontal_flip=True,
                                   validation_split=0.2)

TestData = ImageDataGenerator(rescale=1./255)

LoadTrainData = TrainData.flow_from_directory(
    trainPath,
    target_size=(128, 128),  # Resize all images to 128x128 pixels
    batch_size=32,
    class_mode="binary",     
    subset="training"
)

LoadTestData = TestData.flow_from_directory(
    testPath,
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary",     # Binary classification
)

print(f"Training dataset: {LoadTrainData.samples} images, {LoadTrainData.num_classes} classes")
print(f"Validation dataset: {LoadTestData.samples} images, {LoadTestData.num_classes} classes")

# Print class labels 
print(f"Class labels: {LoadTrainData.class_indices}")
