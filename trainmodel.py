import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
tf.random.set_seed(42)

# Configuration
DATASET_PATH = r"E:\ethics\datasets" 
IMG_SIZE = 224  
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001
MODEL_NAME = "plant_disease_model"

# Paths
TRAIN_DIR = os.path.join(DATASET_PATH, "datasets/train")
VAL_DIR = os.path.join(DATASET_PATH, "datasets/valid")
TEST_DIR = os.path.join(DATASET_PATH, "datasets/test")
MODEL_DIR = os.path.join(os.path.dirname(DATASET_PATH), "models")  # Save models one level up

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Verify dataset structure
print(f"Checking dataset structure...")
print(f"Train directory exists: {os.path.exists(TRAIN_DIR)}")
print(f"Validation directory exists: {os.path.exists(VAL_DIR)}")
print(f"Test directory exists: {os.path.exists(TEST_DIR)}")

# If validation directory doesn't exist with name "valid", try "val"
if not os.path.exists(VAL_DIR):
    VAL_DIR = os.path.join(DATASET_PATH, "val")
    print(f"Trying alternative validation directory: {VAL_DIR}")
    print(f"Alternative validation directory exists: {os.path.exists(VAL_DIR)}")

# Data Augmentation for training
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Minimal preprocessing for validation and test data
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Load and prepare the datasets
print("Loading training data...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

print("Loading validation data...")
validation_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("Loading test data...")
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Get number of classes
num_classes = len(train_generator.class_indices)
print(f"Number of classes: {num_classes}")
print(f"Class names: {list(train_generator.class_indices.keys())}")

# Save class indices for later use
class_indices = train_generator.class_indices
class_indices_inverted = {v: k for k, v in class_indices.items()}

with open(os.path.join(MODEL_DIR, 'class_indices.json'), 'w') as f:
    json.dump(class_indices, f)

# Create the model using transfer learning with MobileNetV2
def create_model():
    # Load the MobileNetV2 model without the top classification layer
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Create a new model on top
    model = Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(), Recall()]
    )
    
    return model

# Create the model
print("Creating model...")
model = create_model()
model.summary()

# Set up callbacks
checkpoint = ModelCheckpoint(
    os.path.join(MODEL_DIR, f"{MODEL_NAME}_best.keras"),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

callbacks = [checkpoint, early_stopping, reduce_lr]

# Calculate steps properly to avoid data exhaustion
train_steps = train_generator.samples // BATCH_SIZE + (1 if train_generator.samples % BATCH_SIZE != 0 else 0)
val_steps = validation_generator.samples // BATCH_SIZE + (1 if validation_generator.samples % BATCH_SIZE != 0 else 0)

# Train the model
print("Starting model training...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    validation_data=validation_generator,
    validation_steps=val_steps,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Save the final model
model.save(os.path.join(MODEL_DIR, f"{MODEL_NAME}_final.keras"))

# Fine-tuning phase
print("\nStarting fine-tuning phase...")
# Unfreeze some layers of the base model
base_model = model.layers[0]
for layer in base_model.layers[-20:]:  # Unfreeze the last 20 layers
    layer.trainable = True

# Recompile with a lower learning rate
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE/10),
    loss='categorical_crossentropy',
    metrics=['accuracy', Precision(), Recall()]
)

# Update checkpoint for fine-tuning
checkpoint_ft = ModelCheckpoint(
    os.path.join(MODEL_DIR, f"{MODEL_NAME}_finetuned.keras"),
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

callbacks_ft = [checkpoint_ft, early_stopping, reduce_lr]

# Continue training with fine-tuning
history_ft = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    validation_data=validation_generator,
    validation_steps=val_steps,
    epochs=5,  # Fewer epochs for fine-tuning
    callbacks=callbacks_ft,
    verbose=1
)

# Combine histories
for key in history.history:
    if key in history_ft.history:
        history.history[key].extend(history_ft.history[key])

# Save the final fine-tuned model
model.save(os.path.join(MODEL_DIR, f"{MODEL_NAME}_final_finetuned.keras"))

# Save the training history
with open(os.path.join(MODEL_DIR, 'training_history.json'), 'w') as f:
    # Convert numpy values to Python types for JSON serialization
    history_dict = {}
    for key, value in history.history.items():
        history_dict[key] = [float(x) for x in value]
    json.dump(history_dict, f)

# Plot training history
plt.figure(figsize=(16, 12))

# Plot training & validation accuracy
plt.subplot(2, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss
plt.subplot(2, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot precision
plt.subplot(2, 2, 3)
if 'precision' in history.history:
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.title('Model Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend(['Train', 'Validation'], loc='upper left')

# Plot recall
plt.subplot(2, 2, 4)
if 'recall' in history.history:
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.title('Model Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
plt.show()

# Calculate test steps properly
test_steps = test_generator.samples // BATCH_SIZE + (1 if test_generator.samples % BATCH_SIZE != 0 else 0)

# Evaluate the model on the test set
print("\nEvaluating model on test set...")
test_loss, test_acc, test_precision, test_recall = model.evaluate(test_generator, steps=test_steps)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")
print(f"Test precision: {test_precision:.4f}")
print(f"Test recall: {test_recall:.4f}")

# Get predictions for the test set
test_generator.reset()
y_pred = model.predict(test_generator, steps=test_steps)
y_pred_classes = np.argmax(y_pred, axis=1)

# Get true labels
y_true = test_generator.classes[:len(y_pred_classes)]

# Generate classification report
class_names = [class_indices_inverted[i] for i in range(num_classes)]
report = classification_report(y_true, y_pred_classes, target_names=class_names, output_dict=True)

# Save classification report
with open(os.path.join(MODEL_DIR, 'classification_report.json'), 'w') as f:
    json.dump(report, f)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# Create confusion matrix
plt.figure(figsize=(20, 20))
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'))
plt.show()

# Create a deployment-ready Python module
with open(os.path.join(MODEL_DIR, 'crop_detection.py'), 'w') as f:
    f.write('''
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class CropDiseaseDetector:
    def _init_(self, model_path=None, class_indices_path=None):
        """Initialize the crop disease detector with a trained model"""
        # Default paths
        if model_path is None:
            model_path = os.path.join('models', 'plant_disease_model_finetuned.keras')
        if class_indices_path is None:
            class_indices_path = os.path.join('models', 'class_indices.json')
        
        # Load the model
        self.model = load_model(model_path)
        print(f"Loaded model from {model_path}")
        
        # Load class indices
        with open(class_indices_path, 'r') as f:
            self.class_indices = json.load(f)
            
        # Invert class indices for prediction
        self.classes = {v: k for k, v in self.class_indices.items()}
        
        # Set image size based on model input
        self.img_size = self.model.input_shape[1]  # Assuming square input
    
    def preprocess_image(self, img_path):
        """Preprocess an image for prediction"""
        # Load and resize image
        img = image.load_img(img_path, target_size=(self.img_size, self.img_size))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess input (same as during training)
        processed_img = preprocess_input(img_array)
        return processed_img
    
    def predict(self, img_path):
        """Predict the disease class for an image"""
        # Preprocess the image
        processed_img = self.preprocess_image(img_path)
        
        # Make prediction
        predictions = self.model.predict(processed_img)
        
        # Get the predicted class index and probability
        pred_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][pred_class_idx])
        
        # Get the class name
        pred_class = self.classes[pred_class_idx]
        
        # Create a dictionary of all probabilities
        all_probabilities = {self.classes[i]: float(prob) for i, prob in enumerate(predictions[0])}
        
        # Return prediction results
        return {
            'class': pred_class,
            'confidence': confidence,
            'all_probabilities': all_probabilities
        }
''')

# Update the disease database in app.py
print("\nUpdating disease database in app.py...")
try:
    # Read the current app.py file
    with open('app.py', 'r') as f:
        app_content = f.read()
    
    # Find the disease_database dictionary
    start_idx = app_content.find('disease_database = {')
    if start_idx != -1:
        # Find the end of the dictionary
        end_idx = app_content.find('}', start_idx)
        end_idx = app_content.find('\n', end_idx)
        
        # Create a new disease database with all classes
        new_db = "disease_database = {\n"
        for class_name in train_generator.class_indices.keys():
            # Format the class name for display
            display_name = class_name.replace('', ' - ').replace('', ' ')
            
            # Create a basic entry for each class
            new_db += f"    '{class_name}': {{\n"
            new_db += f"        'name': '{display_name}',\n"
            new_db += f"        'description': 'A plant disease affecting {class_name.split('')[0].replace('', ' ')}.',\n"
            new_db += f"        'treatment': 'Apply appropriate treatments based on severity. Consult with an agricultural expert.',\n"
            new_db += f"        'prevention': 'Practice crop rotation, maintain plant health, and implement proper sanitation.'\n"
            new_db += f"    }},\n"
        
        new_db += "    # More diseases can be added as needed\n}"
        
        # Replace the old database with the new one
        new_app_content = app_content[:start_idx] + new_db + app_content[end_idx:]
        
        # Write the updated content back to app.py
        with open('app.py', 'w') as f:
            f.write(new_app_content)
        
        print("Successfully updated disease database in app.py")
    else:
        print("Could not find disease_database in app.py. Please update it manually.")
except Exception as e:
    print(f"Error updating app.py: {str(e)}")
    print("Please update the disease_database in app.py manually.")

print("\nTraining complete! Model and supporting files saved to the 'models' directory.")
print("A 'crop_detection.py' file has been created for easy model deployment.")