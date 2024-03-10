import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

training_path = r"D:\projects\optimized hybrid deep grid model\techgium_ui\techgium\techgium_practise\train"
test_path = r"D:\projects\optimized hybrid deep grid model\techgium_ui\techgium\techgium_practise\test"

# Defining the image data generator
train_data = ImageDataGenerator(rescale=1./255,
                                rotation_range=40,
                                horizontal_flip=0.2)

test_data = ImageDataGenerator(rescale=1./255,
                               rotation_range=40,
                               horizontal_flip=0.2)

train_data = train_data.flow_from_directory(directory=training_path,
                                            batch_size=32,
                                            target_size=(200, 200),
                                            class_mode='binary')

test_data = test_data.flow_from_directory(directory=test_path,
                                          batch_size=32,
                                          target_size=(200, 200),
                                          class_mode='binary')

# Defining the model for CNN
def my_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(200, 200, 3)),
        tf.keras.layers.CenterCrop(180, 180),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Defining the hyperparameters
learning_rates = [0.01, 0.001, 0.0001]
momentums = [0.9, 0.95, 0.99]

# Running the SGD optimization with different learning rates and momentums
for lr in learning_rates:
    for momentum in momentums:
        print(f"Learning rate: {lr}, Momentum: {momentum}")
        model = my_model()
        sgd = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
        model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(train_data, epochs=2, validation_data=test_data)
        print(f"Accuracy: {history.history['val_accuracy'][-1]}\n")
