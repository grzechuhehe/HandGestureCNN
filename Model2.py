import numpy as np
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib


# 1. Konfiguracja danych - załadowanie obrazów z katalogów
image_size = (640, 240)  # Rozmiar obrazów (640x240 pikseli)
data_dir = 'Data/leapGestRecog'  # Ścieżka do katalogu z obrazami
test_data_dir = 'Data/TEST'

# ImageDataGenerator do skalowania wartości pikseli i podziału na zbiór treningowy/walidacyjny
data_gen = ImageDataGenerator(
    rescale=1./255,  # Normalizacja pikseli do zakresu [0, 1]
    rotation_range=15,
    width_shift_range=0.125,
    height_shift_range=0.125,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Podział na zbiór walidacyjny (20%)
)

train_data = data_gen.flow_from_directory(
    data_dir,
    target_size=image_size,
    color_mode='grayscale',  # Skala szarości
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = data_gen.flow_from_directory(
    data_dir,
    target_size=image_size,
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Dodanie generatora danych testowych
test_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'Data/TEST',  # Ścieżka do katalogu testowego
    target_size=image_size,
    color_mode='grayscale',
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Nie mieszaj danych, aby zachować kolejność
)

# model 2 : Rozbudowana architektura CNN
model2 = models.Sequential([
    layers.Input(shape=(640, 240, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.425),
    layers.Dense(train_data.num_classes, activation='softmax')
])


# 3. Kompilacja i trening modelu (Model 2)
model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00075),
               loss='categorical_crossentropy',
               metrics=['accuracy'])

history2 = model2.fit(
    train_data,
    validation_data=val_data,
    epochs=40,
    steps_per_epoch=train_data.samples // train_data.batch_size,
    validation_steps=val_data.samples // val_data.batch_size
)

# Wizualizacja wyników dla Modelu 1
plt.figure(figsize=(12, 6))
plt.plot(history2.history['accuracy'], label='Training Accuracy')
plt.plot(history2.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model 2 Accuracy')
plt.savefig('model2_accuracy.png')

plt.figure(figsize=(12, 6))
plt.plot(history2.history['loss'], label='Training Loss')
plt.plot(history2.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model 2 Loss')
plt.savefig('model2_loss.png')


# 4. Ewaluacja modelu 1
loss2, accuracy2 = model2.evaluate(val_data)
print(f'Model 2 - Validation Loss: {loss2}, Validation Accuracy: {accuracy2}')

val_data.reset()
y_pred1 = model2.predict(val_data)
y_pred_classes1 = np.argmax(y_pred1, axis=1)
y_true1 = val_data.classes

print("Classification Report for Model 1")
print(classification_report(y_true1, y_pred_classes1, target_names=list(val_data.class_indices.keys())))

# Ewaluacja modelu 1 na danych testowych
test_loss2, test_accuracy2 = model2.evaluate(test_data)
print(f'Model 1 - Test Loss: {test_loss2}, Test Accuracy: {test_accuracy2}')

test_data.reset()
test_y_pred2 = model2.predict(test_data)
test_y_pred_classes2 = np.argmax(test_y_pred2, axis=1)
test_y_true2 = test_data.classes

print("Test Classification Report for Model 1")
print(classification_report(test_y_true2, test_y_pred_classes2, target_names=list(test_data.class_indices.keys())))

# 5. Zapisanie modeli
model2.save('gesture_recognition_model2.h5')



