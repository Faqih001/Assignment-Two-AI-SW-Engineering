# Original buggy code
import tensorflow as tf
import numpy as np

def create_buggy_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),  # Missing input shape
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Wrong loss function for multi-class classification
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

def generate_dummy_data():
    # Wrong shape for multi-class problem
    X = np.random.rand(100, 784)  # 100 samples, 784 features
    y = np.random.randint(0, 10, 100)  # 10 classes
    return X, y

def main():
    X, y = generate_dummy_data()
    model = create_buggy_model()
    
    # Wrong dimensions, will cause error
    model.fit(X, y, epochs=5, batch_size=32)

if __name__ == "__main__":
    main()
