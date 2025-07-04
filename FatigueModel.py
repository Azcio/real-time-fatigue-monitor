import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from LoadData import LoadTrainData, LoadTestData 

# Build the CNN Model
model = Sequential()

# First Convolutional Layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second Convolutional Layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third Convolutional Layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(1, activation='sigmoid'))

# Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(
    LoadTrainData,
    steps_per_epoch=LoadTrainData.samples // LoadTrainData.batch_size,
    epochs=10,
    validation_data=LoadTestData,
    validation_steps=LoadTestData.samples // LoadTestData.batch_size
)

# Evaluate Model
test_loss, test_acc = model.evaluate(LoadTestData)
print(f"Test accuracy: {test_acc}")

# Save the Model to call it to main.py
model.save('fatigue_detection_model.h5')