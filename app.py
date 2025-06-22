import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="MNIST Digit Classifier", page_icon="✏️")

def load_model():
    try:
        model = tf.keras.models.load_model('mnist_model')
        return model
    except:
        st.error("Model not found. Please make sure to train and save the model first.")
        return None

def predict_digit(image, model):
    # Preprocess the image
    image = image.resize((28, 28))
    image = image.convert('L')  # Convert to grayscale
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1)
    image = image / 255.0
    
    # Make prediction
    prediction = model.predict(image)
    return np.argmax(prediction[0]), prediction[0]

def main():
    st.title("✏️ MNIST Digit Classifier")
    st.markdown("""
    ### Draw a digit (0-9) in the canvas below
    Try to center your digit and make it large enough to fill most of the canvas.
    """)
    
    model = load_model()
    
    if model is None:
        return
        
    # Create a canvas component
    canvas_result = st_canvas(
        stroke_width=20,
        stroke_color='white',
        background_color='black',
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if canvas_result.image_data is not None:
            # Convert the drawn image to PIL Image
            image = Image.fromarray(canvas_result.image_data.astype('uint8'))
            if st.button('Predict', type='primary'):
                digit, probabilities = predict_digit(image, model)
                
                st.markdown(f"### Predicted Digit: {digit}")
                
                # Display probabilities as a bar chart
                st.markdown("#### Prediction Probabilities")
                prob_df = pd.DataFrame({
                    'Digit': range(10),
                    'Probability': probabilities
                })
                st.bar_chart(prob_df.set_index('Digit'))
    
    with col2:
        st.markdown("### Instructions")
        st.markdown("""
        1. Draw a single digit
        2. Click 'Predict'
        3. See results
        4. Clear canvas to try again
        """)

if __name__ == "__main__":
    main()
