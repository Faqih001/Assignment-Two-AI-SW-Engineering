import os
import logging

# Configure logging to suppress TensorFlow warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Rest of the imports
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
    
    # File upload for code with bugs
    st.subheader("Upload Code to Fix")
    uploaded_file = st.file_uploader("Choose a Python file", type="py")
    
    if uploaded_file is not None:
        # Read and display the uploaded code
        content = uploaded_file.getvalue().decode()
        st.subheader("Original Code")
        st.code(content, language='python')
        
        # Analyze and fix the code
        fixes = analyze_code(content)
        
        # Display fixes
        st.subheader("Suggested Fixes")
        for i, (issue, fix) in enumerate(fixes.items(), 1):
            st.markdown(f"**Issue {i}**: {issue}")
            st.markdown(f"**Fix**: {fix}")
        
        # Show example of fixed code
        st.subheader("Example of Fixed Code")
        fixed_code = apply_fixes(content)
        st.code(fixed_code, language='python')
        
        # Option to download fixed code
        st.download_button(
            label="Download Fixed Code",
            data=fixed_code,
            file_name="fixed_" + uploaded_file.name,
            mime="text/plain"
        )
    
    # Show example buggy code
    st.subheader("Example Buggy Code")
    with open('buggy_model.py', 'r') as file:
        buggy_code = file.read()
        st.code(buggy_code, language='python')
    
    # Show example fixed code
    st.subheader("Example Fixed Code")
    with open('fixed_model.py', 'r') as file:
        fixed_code = file.read()
        st.code(fixed_code, language='python')
    
    # Common TensorFlow/Keras bugs and fixes
    st.subheader("Common TensorFlow/Keras Bugs and Fixes")
    st.markdown("""
    1. **Missing Input Shape**
       - Bug: First layer doesn't specify input shape
       - Fix: Add input_shape parameter to first layer
       
    2. **Wrong Loss Function**
       - Bug: Using binary_crossentropy for multi-class problems
       - Fix: Use sparse_categorical_crossentropy or categorical_crossentropy
       
    3. **Data Preprocessing**
       - Bug: Missing data normalization
       - Fix: Normalize data (e.g., divide by 255 for images)
       
    4. **Model Validation**
       - Bug: No validation split during training
       - Fix: Add validation_split or validation_data
       
    5. **Data Shape Issues**
       - Bug: Incorrect input dimensions
       - Fix: Reshape data to match expected input shape
    """)

def analyze_code(content):
    """Analyze code for common bugs"""
    fixes = {}
    
    # Check for common issues
    if 'input_shape' not in content and 'Dense' in content:
        fixes["Missing Input Shape"] = "Add input_shape parameter to the first layer"
    
    if 'binary_crossentropy' in content and 'Dense' in content and 'softmax' in content:
        fixes["Wrong Loss Function"] = "Use sparse_categorical_crossentropy for multi-class problems"
    
    if '/255' not in content and ('mnist' in content or 'image' in content.lower()):
        fixes["Missing Normalization"] = "Normalize pixel values by dividing by 255"
    
    if 'validation_split' not in content and 'fit' in content:
        fixes["No Validation"] = "Add validation_split parameter to model.fit()"
    
    if 'reshape' not in content and ('mnist' in content or 'Dense' in content):
        fixes["Data Shape Issues"] = "Reshape input data to match the expected shape"
    
    return fixes

def apply_fixes(content):
    """Apply fixes to the code"""
    fixed_code = content
    
    # Example fixes (you can expand these based on common patterns)
    if 'input_shape' not in content and 'Dense' in content:
        fixed_code = fixed_code.replace(
            'Dense(128',
            'Dense(128, input_shape=(784,)'
        )
    
    if 'binary_crossentropy' in content:
        fixed_code = fixed_code.replace(
            'binary_crossentropy',
            'sparse_categorical_crossentropy'
        )
    
    if 'validation_split' not in content and 'fit' in content:
        fixed_code = fixed_code.replace(
            'model.fit(',
            'model.fit(validation_split=0.2,'
        )
    
    return fixed_code

def main():
    # Initialize session states
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = "MNIST Classifier"
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'username' not in st.session_state:
        st.session_state['username'] = None

    if not st.session_state['logged_in']:
        # Show login page
        show_login_page()
        
        # Check if login was successful
        if st.session_state.get('logged_in', False):
            st.session_state['current_page'] = "MNIST Classifier"
            st.experimental_set_query_params(page="MNIST Classifier")
    else:
        st.sidebar.title(f"Welcome, {st.session_state['username']}!")
        st.sidebar.title("Navigation")
        
        # Navigation
        pages = ["MNIST Classifier", "Iris Classifier", "NLP Analysis", "Bug Fix Demo"]
        current_page_idx = pages.index(st.session_state['current_page'])
        
        selected_page = st.sidebar.radio(
            "Go to",
            pages,
            index=current_page_idx
        )
        
        # Update page if changed
        if selected_page != st.session_state['current_page']:
            st.session_state['current_page'] = selected_page
            st.experimental_set_query_params(page=selected_page)
        
        # Logout button
        if st.sidebar.button("Logout"):
            for key in ['logged_in', 'username', 'current_page']:
                if key in st.session_state:
                    del st.session_state[key]
            st.experimental_set_query_params()
            return
        
        # Page routing
        if st.session_state['current_page'] == "MNIST Classifier":
            mnist_classifier()
        elif st.session_state['current_page'] == "Iris Classifier":
            iris_classifier()
        elif st.session_state['current_page'] == "NLP Analysis":
            nlp_analysis()
        elif st.session_state['current_page'] == "Bug Fix Demo":
            bug_fix_demo()

if __name__ == "__main__":
    main()