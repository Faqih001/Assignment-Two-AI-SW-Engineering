import tensorflow as tf
import numpy as np

def create_fixed_model():
    model = tf.keras.Sequential([
        # Added input shape for first layer
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Changed to correct loss function for multi-class classification
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def generate_dummy_data():
    # Correct shape for multi-class problem
    X = np.random.rand(100, 784)  # 100 samples, 784 features
    y = np.random.randint(0, 10, 100)  # 10 classes, one label per sample
    return X, y

def main():
    X, y = generate_dummy_data()
    model = create_fixed_model()
    
    # Added validation split
    history = model.fit(X, y, 
                       epochs=5, 
                       batch_size=32,
                       validation_split=0.2,
                       verbose=1)
    return history

if __name__ == "__main__":
    main()
