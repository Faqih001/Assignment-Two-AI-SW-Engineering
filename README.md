# AI Tools Assignment: Mastering the AI Toolkit 🛠️🧠

This repository contains the implementation of various AI tasks using different frameworks and tools. The assignment is divided into three main parts, focusing on theoretical understanding, practical implementation, and ethical considerations in AI development.

## Part 1: Theoretical Understanding

### Q1: TensorFlow vs PyTorch Comparison

**Key Differences:**
- **TensorFlow:**
  - Static computational graphs (Graph mode)
  - Better production deployment tools (TF Serving, TFLite)
  - Stronger enterprise support
  - Excellent for production systems

- **PyTorch:**
  - Dynamic computational graphs
  - More Pythonic and intuitive
  - Easier debugging
  - Great for research

**When to Choose:**
- Use TensorFlow for:
  - Production deployment
  - Mobile applications
  - Enterprise solutions
  - TensorFlow Extended (TFX) pipelines

- Use PyTorch for:
  - Research projects
  - Rapid prototyping
  - Dynamic neural networks
  - Academic work

### Q2: Jupyter Notebooks Use Cases

1. **Data Analysis & Visualization**
   - Interactive data exploration
   - Real-time visualization
   - Quick iterations on data cleaning
   - Immediate feedback on analysis

2. **Model Development**
   - Step-by-step model building
   - Visual training progress
   - Interactive hyperparameter tuning
   - Result documentation

### Q3: spaCy vs Basic Python String Operations

spaCy provides several advantages:
1. **Pre-trained Models**
   - Statistical models ready for use
   - Multiple language support
   - Regular updates and improvements

2. **Advanced NLP Features**
   - Named Entity Recognition
   - POS tagging
   - Dependency parsing
   - Word vectors

3. **Performance Benefits**
   - Cython optimization
   - Efficient memory usage
   - Fast processing

## Project Structure

```plaintext
.
├── notebooks/
│   └── ai_tools_assignment.ipynb
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Clone this repository

2. Create a Python virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Download spaCy model:

   ```bash
   python -m spacy download en_core_web_sm
   ```

## Running the Notebook

1. Start Jupyter:

   ```bash
   jupyter notebook
   ```

2. Open `notebooks/ai_tools_assignment.ipynb`

## Running the Streamlit App

To run the MNIST classifier web interface:

```bash
streamlit run app.py
```

## Tasks Overview

### 1. Classical ML with Scikit-learn

- Iris Species Classification using Decision Tree
- Data preprocessing and model evaluation

### 2. Deep Learning with TensorFlow

- MNIST Digit Classification using CNN
- Model training and visualization

### 3. NLP with spaCy

- Named Entity Recognition
- Sentiment Analysis on Amazon Reviews

### Bonus: Web Deployment

- Interactive web interface using Streamlit
- Real-time digit classification

## Evaluation Criteria

- Theoretical Accuracy (30%)
- Code Functionality & Quality (40%)
- Ethical Analysis (15%)
- Creativity & Presentation (15%)

## Contributors

[Add team member names here]

## License

[Add license information]

## Part 2: Practical Implementation

### MNIST Classifier Results
Our TensorFlow-based CNN model achieved:
- Test accuracy: 99.18%
- Test loss: 0.0313
- Training time: ~10 minutes (CPU)

### Model Architecture
```python
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

### Training Process
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Epochs: 10
- Batch size: 32
- Validation split: 20%

# Completed Tasks and Results

## Part 1: Theoretical Understanding

### 1. Short Answer Questions

**Q1: TensorFlow vs PyTorch Differences**
- **TensorFlow Strengths:**
  - Static computational graphs
  - Better production deployment (TF Serving)
  - Enterprise-grade tools
  - Mobile deployment (TFLite)

- **PyTorch Strengths:**
  - Dynamic computational graphs
  - More Pythonic syntax
  - Better debugging
  - Research-friendly

**Choice Criteria:**
- Choose TensorFlow for production deployment, mobile apps
- Choose PyTorch for research, prototyping, education

**Q2: Jupyter Notebooks Use Cases**
1. **Data Analysis & Exploration:**
   - Interactive visualization
   - Real-time data manipulation
   - Step-by-step analysis
   - Documentation inline with code

2. **Model Development:**
   - Iterative model building
   - Training visualization
   - Interactive debugging
   - Result documentation

**Q3: spaCy vs Basic String Operations**
- **Advanced Features:**
  - Named Entity Recognition
  - Part-of-speech tagging
  - Dependency parsing
  - Pre-trained models

- **Performance Benefits:**
  - Optimized C/Cython implementation
  - Efficient memory usage
  - Fast processing pipeline

## Part 2: Practical Implementation

### Task 1: Iris Classification
- Implemented in `iris_classifier.py`
- Features:
  - Data preprocessing
  - Decision Tree classifier
  - Feature importance visualization
  - Interactive prediction interface

### Task 2: MNIST Classification
- Implemented in `train_model.py` and `app.py`
- Results:
  - Test accuracy: 99.18%
  - Test loss: 0.0313
  - Interactive web interface

### Task 3: NLP Analysis
- Implemented in `nlp_analysis.py`
- Features:
  - Named Entity Recognition
  - Sentiment analysis
  - POS tagging
  - Interactive text analysis

## Part 3: Ethics & Optimization

### 1. Ethical Considerations
- **Data Bias Mitigation:**
  - Diverse training data
  - Regular bias testing
  - User feedback incorporation
  - Transparent model decisions

### 2. Bug Fix Demo
- Created `buggy_model.py` and `fixed_model.py`
- Fixed issues:
  1. Missing input shape
  2. Incorrect loss function
  3. Added validation split

## Bonus Task: Web Application

### Features
1. **Security**
   - Login system
   - Session management
   - Secure password handling

2. **Navigation**
   - Sidebar menu
   - Multiple task pages
   - Logout functionality

3. **Interactive Components**
   - Drawing canvas for MNIST
   - Sliders for Iris prediction
   - Text input for NLP analysis

### Pages
1. **MNIST Classifier**
   - Drawing interface
   - Real-time prediction
   - Confidence visualization

2. **Iris Classifier**
   - Interactive feature input
   - Model training option
   - Feature importance plots

3. **NLP Analysis**
   - Text input
   - Entity visualization
   - POS tag display

### Usage Instructions

1. **Setup**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate

   # Install requirements
   pip install -r requirements.txt

   # Download spaCy model
   python -m spacy download en_core_web_sm
   ```

2. **Run the Application**
   ```bash
   streamlit run app.py
   ```

3. **Login Credentials**
   - Username: demo
   - Password: password123

### Deployment Notes

1. **Local Deployment**
   - Runs on http://localhost:8501
   - Requires Python 3.8+
   - GPU optional

2. **Remote Deployment**
   - Push to GitHub
   - Connect to Streamlit Cloud
   - Configure environment
   - Set up secrets

## Model Training Script (train_model.py)

Our `train_model.py` implements a CNN for MNIST digit classification with the following key components:

### Data Preprocessing

- Loads MNIST dataset using TensorFlow
- Normalizes pixel values to [0, 1] range
- Reshapes images to (28, 28, 1) for CNN input
- Splits data into training and test sets

### CNN Architecture Details

```python
model = models.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Third Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    
    # Flatten and Dense layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])
```

### Key Features

1. **Advanced Architecture**
   - Three convolutional blocks
   - Batch normalization for stability
   - Dropout for regularization
   - MaxPooling for dimension reduction

2. **Training Configuration**
   - Optimizer: Adam
   - Loss: Sparse Categorical Crossentropy
   - Batch size: 64
   - Epochs: 10
   - Validation split: 20%

3. **Visualization**
   - Training/validation accuracy plots
   - Training/validation loss plots
   - Saves plots as 'training_history.png'

### Usage

```bash
python train_model.py
```

The script will perform the following steps:

1. Load and preprocess the MNIST dataset
2. Create and compile the model
3. Train for 10 epochs
4. Generate training visualizations
5. Save the trained model as 'mnist_model.keras'

### Performance Optimizations

- BatchNormalization layers for faster training
- Dropout layer (0.5) for regularization
- Efficient data preprocessing pipeline
- GPU support when available

## User Authentication System

The application includes a secure user authentication system with the following features:

### Authentication Features

- User registration with email verification
- Secure password storage using SHA-256 hashing
- Session management
- User-specific data persistence

### User Management

1. **Create Account**
   - Click on the "Sign Up" tab
   - Enter username, password, and email
   - System validates and creates account

2. **Login**
   - Enter username and password
   - Access all application features
   - Session persists until logout

3. **Security Features**
   - Passwords are hashed before storage
   - Session-based authentication
   - CSRF protection via Streamlit
   - Input validation and sanitization

### Authentication Components

- `users.py`: User management system
- `login.py`: Login/signup interface
- `app.py`: Main application with authentication integration

### User Data Storage

User data is stored securely in `users.json` with the following structure:

```json
{
    "username": {
        "password": "hashed_password",
        "email": "user@example.com"
    }
}
```

## Cloud Deployment

### Streamlit Cloud Deployment

1. **Prerequisites**
   - GitHub account
   - Streamlit account (sign up at https://share.streamlit.io)
   - All code pushed to a public GitHub repository

2. **Deployment Steps**

   a. **Prepare Repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

   b. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io
   - Sign in with GitHub
   - Select your repository
   - Select `app.py` as the main file
   - Click "Deploy"

3. **Environment Setup**
   - The `requirements.txt` file contains all necessary dependencies
   - Using `tensorflow-cpu` to reduce cloud resource usage
   - Configured for headless OpenCV

4. **Security Considerations**
   - User data is stored in `users.json` (in production, use a proper database)
   - Configure Streamlit secrets for sensitive information
   - HTTPS enabled by default on Streamlit Cloud

5. **Monitoring**
   - View application metrics in Streamlit Cloud dashboard
   - Check deployment logs for issues
   - Monitor resource usage

### Post-Deployment Tasks

1. **Testing**
   - Verify all features work in cloud environment
   - Test user registration and login
   - Validate model predictions
   - Check file upload/download functionality

2. **Maintenance**
   - Regularly update dependencies
   - Monitor error logs
   - Backup user data
   - Update model weights as needed

3. **Scaling**
   - Monitor resource usage
   - Optimize model loading
   - Cache frequent operations
   - Use efficient data storage

### Production Considerations

1. **Data Persistence**
   For production deployment, consider:
   - Using a proper database (e.g., PostgreSQL)
   - Setting up cloud storage for models
   - Implementing proper backup strategies

2. **Security**
   - Implement rate limiting
   - Add request validation
   - Set up monitoring alerts
   - Regular security audits

3. **Performance**
   - Cache model predictions
   - Optimize image processing
   - Minimize memory usage
   - Load models efficiently
