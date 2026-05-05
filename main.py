import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam

# ========================
# DATASET PATH (UPDATED)
# ========================
data_dir = os.path.join(
    "lung_dataset",
    "The IQ-OTHNCCD lung cancer dataset"
)
# Get only folders (classes)
classes = [cls for cls in os.listdir(data_dir)
           if os.path.isdir(os.path.join(data_dir, cls))]
print("Classes:", classes)

# ========================
# PREPARE TRAIN/VAL SPLIT
# ========================
base_dir = "processed_data"
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
for cls in classes:
    class_path = os.path.join(data_dir, cls)
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
    images = [
        os.path.join(class_path, img)
        for img in os.listdir(class_path)
        if img.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    print(f"{cls}: {len(images)} images")
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
    for img_path in train_imgs:
        shutil.copy(img_path, os.path.join(train_dir, cls, os.path.basename(img_path)))

    for img_path in val_imgs:
        shutil.copy(img_path, os.path.join(val_dir, cls, os.path.basename(img_path)))

# ========================
# DATA GENERATORS
# ========================
IMG_SIZE = 299
BATCH_SIZE = 16
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)
val_gen = ImageDataGenerator(rescale=1./255)
train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ========================
# MODEL
# ========================
base_model = Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
for layer in base_model.layers[:-20]:
    layer.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(len(train_data.class_indices), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ========================
# TRAIN
# ========================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=8
)

# ========================
# SAVE MODEL
# ========================
model.save("lung_cancer_xception.keras")
print("✅ Model saved successfully!")

# ========================
# PLOTS
# ========================
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.legend(["Train", "Validation"])
plt.show()
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")
plt.legend(["Train", "Validation"])
plt.show()

# ========================
# EVALUATION
# ========================
val_preds = model.predict(val_data)
val_preds_classes = np.argmax(val_preds, axis=1)
print("\nClassification Report:\n")
print(classification_report(val_data.classes, val_preds_classes))
cm = confusion_matrix(val_data.classes, val_preds_classes)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=train_data.class_indices.keys(),
            yticklabels=train_data.class_indices.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


