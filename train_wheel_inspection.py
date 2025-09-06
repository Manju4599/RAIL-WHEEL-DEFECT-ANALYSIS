# train_wheel_inspection.py
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import keras
layers = keras.layers
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class SyntheticWheelDataGenerator:
    """Generate synthetic wheel images for training and testing"""
    
    def __init__(self, img_size=(128, 128)):
        self.img_size = img_size
        self.healthy_templates = self._create_healthy_templates()
        
    def _create_healthy_templates(self):
        """Create base templates for healthy wheels from different angles"""
        templates = {}
        
        # Outer view template (circular shape)
        outer = np.zeros(self.img_size, dtype=np.uint8)
        cv2.circle(outer, (self.img_size[0]//2, self.img_size[1]//2), 
                  self.img_size[0]//3, 255, -1)
        templates['outer'] = outer
        
        # Runway view template (elliptical shape)
        runway = np.zeros(self.img_size, dtype=np.uint8)
        cv2.ellipse(runway, (self.img_size[0]//2, self.img_size[1]//2), 
                   (self.img_size[0]//2, self.img_size[1]//4), 0, 0, 360, 255, -1)
        templates['runway'] = runway
        
        # Inside track template (complex shape)
        inside = np.zeros(self.img_size, dtype=np.uint8)
        center = (self.img_size[0]//2, self.img_size[1]//2)
        radius = self.img_size[0]//3
        cv2.circle(inside, center, radius, 255, -1)
        # Add some internal structure
        cv2.circle(inside, center, radius//2, 0, -1)
        templates['inside'] = inside
        
        return templates
    
    def _add_defects(self, image, defect_type):
        """Add different types of defects to wheel images"""
        img = image.copy()
        h, w = img.shape
        
        if defect_type == 'crack':
            # Add crack-like lines
            start_point = (np.random.randint(w//4, 3*w//4), np.random.randint(h//4, 3*h//4))
            end_point = (start_point[0] + np.random.randint(-20, 20), 
                        start_point[1] + np.random.randint(-20, 20))
            cv2.line(img, start_point, end_point, 0, np.random.randint(2, 5))
            
        elif defect_type == 'flat_spot':
            # Create a flat spot on the wheel
            center = (w//2, h//2)
            angle = np.random.randint(0, 360)
            start_angle = angle - 15
            end_angle = angle + 15
            cv2.ellipse(img, center, (w//3, h//3), 0, start_angle, end_angle, 0, -1)
            
        elif defect_type == 'pitting':
            # Add small pits/craters
            for _ in range(np.random.randint(5, 15)):
                x = np.random.randint(w//4, 3*w//4)
                y = np.random.randint(h//4, 3*h//4)
                radius = np.random.randint(2, 6)
                cv2.circle(img, (x, y), radius, 0, -1)
                
        elif defect_type == 'wear':
            # Simulate uneven wear
            center = (w//2, h//2)
            radius = w//3 - np.random.randint(5, 15)
            cv2.circle(img, center, radius, 255, -1)
            
        return img
    
    def generate_sample(self, defect_type=None):
        """Generate a sample with three views"""
        views = {}
        
        for view_name in ['outer', 'runway', 'inside']:
            # Start with healthy template
            img = self.healthy_templates[view_name].copy()
            
            # Add noise to simulate real conditions
            noise = np.random.normal(0, 25, img.shape).astype(np.int32)
            img = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)
            
            # Add defects if specified
            if defect_type:
                img = self._add_defects(img, defect_type)
                
            # Normalize and add channel dimension
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=-1)
            
            views[view_name] = img
            
        return views
    
    def generate_dataset(self, n_samples_per_class=500):
        """Generate a complete dataset"""
        X_outer, X_runway, X_inside = [], [], []
        y = []
        
        defect_types = [None, 'crack', 'flat_spot', 'pitting', 'wear']
        
        for defect_idx, defect_type in enumerate(defect_types):
            for _ in range(n_samples_per_class):
                sample = self.generate_sample(defect_type)
                X_outer.append(sample['outer'])
                X_runway.append(sample['runway'])
                X_inside.append(sample['inside'])
                y.append(0 if defect_type is None else 1)  # Binary classification: 0=healthy, 1=defective
                
        return (np.array(X_outer), np.array(X_runway), np.array(X_inside)), np.array(y)

def create_multi_view_model(input_shape=(128, 128, 1)):
    """Create a multi-view CNN model"""
    # Input layers for each view
    outer_input = layers.Input(shape=input_shape, name='outer_view')
    runway_input = layers.Input(shape=input_shape, name='runway_view')
    inside_input = layers.Input(shape=input_shape, name='inside_view')
    
    # Base CNN architecture for feature extraction
    def create_cnn_branch(input_layer):
        x = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        return x
    
    # Process each view through the CNN
    outer_features = create_cnn_branch(outer_input)
    runway_features = create_cnn_branch(runway_input)
    inside_features = create_cnn_branch(inside_input)
    
    # Concatenate features from all views
    merged = layers.concatenate([outer_features, runway_features, inside_features])
    
    # Additional layers for classification
    x = layers.Dense(128, activation='relu')(merged)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = keras.Model(
        inputs=[outer_input, runway_input, inside_input],
        outputs=output,
        name='wheel_inspection_model'
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """Compile and train the model"""
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
    
    # Train the model
    history = model.fit(
        [X_train[0], X_train[1], X_train[2]],
        y_train,
        batch_size=32,
        epochs=50,
        validation_data=([X_val[0], X_val[1], X_val[2]], y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and plot results"""
    # Predictions
    y_pred = model.predict([X_test[0], X_test[1], X_test[2]])
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred_binary))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_binary)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return y_pred

def visualize_predictions(model, X_test, y_test, generator, n_samples=5):
    """Visualize model predictions with Grad-CAM"""
    # Select random samples
    indices = np.random.choice(len(X_test[0]), n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        # Get the sample and prediction
        sample = [X_test[0][idx:idx+1], X_test[1][idx:idx+1], X_test[2][idx:idx+1]]
        prediction = model.predict(sample)[0][0]
        true_label = y_test[idx]
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        views = ['Outer View', 'Runway View', 'Inside View']
        
        for j, (view_name, ax) in enumerate(zip(views, axes)):
            # Display the image
            img = sample[j][0].squeeze()
            ax.imshow(img, cmap='gray')
            ax.set_title(f'{view_name}')
            ax.axis('off')
        
        # Set overall title with prediction
        fig.suptitle(f'True: {"Defective" if true_label else "Healthy"} | '
                    f'Predicted: {"Defective" if prediction > 0.5 else "Healthy"} '
                    f'({prediction:.3f})', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(f'prediction_{i}.png')
        plt.close()

def main():
    """Main function to run the entire pipeline"""
    print("Generating synthetic dataset...")
    generator = SyntheticWheelDataGenerator()
    X, y = generator.generate_dataset(n_samples_per_class=200)
    
    # Split the dataset
    X_outer, X_runway, X_inside = X
    X_outer_train_val, X_outer_test, X_runway_train_val, X_runway_test, X_inside_train_val, X_inside_test, y_train_val, y_test = train_test_split(
        X_outer, X_runway, X_inside, y, test_size=0.2, random_state=42, stratify=y
    )

    X_outer_train, X_outer_val, X_runway_train, X_runway_val, X_inside_train, X_inside_val, y_train, y_val = train_test_split(
        X_outer_train_val, X_runway_train_val, X_inside_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )

    X_train = [X_outer_train, X_runway_train, X_inside_train]
    X_val = [X_outer_val, X_runway_val, X_inside_val]
    X_test = [X_outer_test, X_runway_test, X_inside_test]

    print(f"Training set size: {len(y_train)}")
    print(f"Validation set size: {len(y_val)}")
    print(f"Test set size: {len(y_test)}")
    
    # Create and train model
    print("Creating model...")
    model = create_multi_view_model()
    model.summary()
    
    print("Training model...")
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)
    
    # Visualize some predictions
    print("Generating visualizations...")
    visualize_predictions(model, X_test, y_test, generator)
    
    # Save the model
    model.save('wheel_inspection_model.h5')
    print("Model saved as wheel_inspection_model.h5")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    print("All done! Check the generated images and model file.")

if __name__ == "__main__":
    main()