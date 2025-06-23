import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2

def load_model():
    """Load the trained MNIST model"""
    try:
        model = tf.keras.models.load_model('mnist_model.keras')
        return model
    except:
        st.error("Model not found. Please train the model first by running train_model.py")
        return None

def preprocess_image(image):
    """Preprocess the drawn image for prediction"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to 28x28
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values
    normalized = resized.astype('float32') / 255.0
    
    # Reshape for model input
    reshaped = normalized.reshape(1, 28, 28, 1)
    
    return reshaped

def main():
    st.title("MNIST Digit Classifier")
    st.write("Draw a digit (0-9) below and the model will predict what digit it is!")
    
    # Create a canvas for drawing
    canvas_result = st_canvas(
        stroke_width=20,
        stroke_color='#FFFFFF',
        background_color='#000000',
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )
    
    # Load the model
    model = load_model()
    
    if canvas_result.image_data is not None and model is not None:
        # Get the drawn image
        image = canvas_result.image_data
        
        # Only predict if something is drawn
        if image.sum() > 0:
            # Preprocess the image
            processed_image = preprocess_image(image)
            
            # Make prediction
            prediction = model.predict(processed_image)
            predicted_digit = np.argmax(prediction[0])
            confidence = prediction[0][predicted_digit] * 100
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Prediction")
                st.title(f"{predicted_digit}")
            with col2:
                st.subheader("Confidence")
                st.title(f"{confidence:.2f}%")
            
            # Display bar chart of probabilities
            st.bar_chart(prediction[0])

if __name__ == "__main__":
    main()