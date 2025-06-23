import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import spacy
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import plotly.express as px
import hashlib

# Import user management
from login import show_login_page

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = None

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

def mnist_classifier():
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

def iris_classifier():
    st.title("Iris Species Classification")
    
    # Load and preprocess data
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    
    # Train model
    if st.button("Train Model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        st.success(f"Model trained! Accuracy: {score:.2f}")
        
        # Feature importance plot
        importances = pd.DataFrame({
            'feature': iris.feature_names,
            'importance': model.feature_importances_
        })
        fig = px.bar(importances, x='feature', y='importance', 
                    title='Feature Importance')
        st.plotly_chart(fig)
    
    # Interactive prediction
    st.subheader("Make a Prediction")
    input_features = {}
    for feature in iris.feature_names:
        input_features[feature] = st.slider(feature, 0.0, 10.0, 5.0)
    
    if st.button("Predict"):
        X_pred = pd.DataFrame([input_features])
        model = DecisionTreeClassifier()
        model.fit(X, y)
        prediction = model.predict(X_pred)
        st.success(f"Predicted Species: {iris.target_names[prediction[0]]}")

def nlp_analysis():
    st.title("NLP Analysis with spaCy")
    
    # Text input
    text = st.text_area("Enter text for analysis", 
                        "I love my new Apple iPhone! The camera quality is amazing.")
    
    if st.button("Analyze"):
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            st.error("Please install spaCy model using: python -m spacy download en_core_web_sm")
            return
        
        # Process text
        doc = nlp(text)
        
        # Display entities
        st.subheader("Named Entities")
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        if entities:
            df_entities = pd.DataFrame(entities, columns=['Text', 'Entity Type'])
            st.table(df_entities)
        else:
            st.write("No entities found")
        
        # Display tokens and POS tags
        st.subheader("Tokens and Part-of-Speech Tags")
        tokens = [(token.text, token.pos_) for token in doc]
        df_tokens = pd.DataFrame(tokens, columns=['Token', 'POS'])
        st.table(df_tokens)

def bug_fix_demo():
    st.title("Bug Fix Demonstration")
    
    st.subheader("Original Buggy Code")
    with open('buggy_model.py', 'r') as file:
        st.code(file.read(), language='python')
    
    st.subheader("Fixed Code")
    with open('fixed_model.py', 'r') as file:
        st.code(file.read(), language='python')
    
    st.markdown("""
    ### Bug Fixes Explained
    1. **Input Shape**: Added missing input_shape parameter in the first Dense layer
    2. **Loss Function**: Changed from binary_crossentropy to sparse_categorical_crossentropy
    3. **Model Evaluation**: Added validation split for proper evaluation
    """)

def main():
    if not st.session_state['logged_in']:
        show_login_page()
    else:
        st.sidebar.title(f"Welcome, {st.session_state['username']}!")
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Go to", [
            "MNIST Classifier",
            "Iris Classifier",
            "NLP Analysis",
            "Bug Fix Demo"
        ])
        
        if st.sidebar.button("Logout"):
            st.session_state['logged_in'] = False
            st.experimental_rerun()
        
        if page == "MNIST Classifier":
            mnist_classifier()
        elif page == "Iris Classifier":
            iris_classifier()
        elif page == "NLP Analysis":
            nlp_analysis()
        elif page == "Bug Fix Demo":
            bug_fix_demo()

if __name__ == "__main__":
    main()