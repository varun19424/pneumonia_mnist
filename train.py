import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations



import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import class_weight
import medmnist
from medmnist import PneumoniaMNIST

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load PneumoniaMNIST dataset
def load_data():
    train_dataset = PneumoniaMNIST(split='train', download=True, size=28)
    val_dataset = PneumoniaMNIST(split='val', download=True, size=28)
    test_dataset = PneumoniaMNIST(split='test', download=True, size=28)
    
    X_train, y_train = train_dataset.imgs, train_dataset.labels
    X_val, y_val = val_dataset.imgs, val_dataset.labels
    X_test, y_test = test_dataset.imgs, test_dataset.labels
    
    # Normalize pixel values
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# Data augmentation
def create_data_generators():
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator()
    return train_datagen, val_datagen

# Build and fine-tune Inception-V3 model
def build_model():
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    
    # Freeze base model layers
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                 loss='binary_crossentropy',
                 metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model

# Compute class weights to handle imbalance
def compute_class_weights(y_train):
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train.flatten()), y=y_train.flatten())
    return dict(enumerate(class_weights))

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    return accuracy, precision, recall, f1

# Main training function
def main():
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Data generators
    train_datagen, val_datagen = create_data_generators()
    
    # Convert grayscale to RGB
    X_train_rgb = np.repeat(X_train[..., np.newaxis], 3, -1)
    X_val_rgb = np.repeat(X_val[..., np.newaxis], 3, -1)
    X_test_rgb = np.repeat(X_test[..., np.newaxis], 3, -1)
    
    # Resize to (299, 299, 3)
    X_train_resized = np.array([tf.image.resize(img, (299, 299)).numpy() for img in X_train_rgb])
    X_val_resized = np.array([tf.image.resize(img, (299, 299)).numpy() for img in X_val_rgb])
    X_test_resized = np.array([tf.image.resize(img, (299, 299)).numpy() for img in X_test_rgb])
    
    # Create data generators
    train_generator = train_datagen.flow(X_train_resized, y_train, batch_size=32)
    val_generator = val_datagen.flow(X_val_resized, y_val, batch_size=32)
    
    # Build model
    model = build_model()
    
    # Class weights
    class_weights = compute_class_weights(y_train)
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        class_weight=class_weights,
        callbacks=[early_stopping, lr_reducer]
    )
    
    # Evaluate model
    evaluate_model(model, X_test_resized, y_test)
    
    # Save model
    model.save('pneumonia_inceptionv3.h5')

if __name__ == '__main__':
    main()