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

## Part 3: Ethics & Optimization

### Ethical Considerations
1. **Data Representation**
   - Limited handwriting styles
   - Cultural variations
   - Accessibility concerns

2. **Mitigation Strategies**
   - Diverse data collection
   - Regular bias testing
   - User feedback incorporation

### Performance Optimization
1. **Training Optimizations**
   - Batch normalization
   - Early stopping
   - Learning rate scheduling

2. **Deployment Optimizations**
   - Model quantization
   - CPU optimizations
   - Memory management

## Deployment Guide

### Local Deployment
1. Clone the repository
2. Set up virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Train the model:
   ```bash
   python train_model.py
   ```
5. Run Streamlit app:
   ```bash
   streamlit run app.py
   ```

### Remote Deployment (Streamlit Cloud)
1. Create a GitHub repository
2. Push your code with:
   - requirements.txt
   - .streamlit/config.toml
   - Proper directory structure
3. Connect to Streamlit Cloud:
   - Sign up at share.streamlit.io
   - Connect your repository
   - Configure deployment settings
4. Monitor performance:
   - Check app metrics
   - Monitor resource usage
   - Track user interactions

### Future Improvements
1. **Model Enhancements**
   - Data augmentation
   - Transfer learning
   - Model ensembles

2. **Application Features**
   - Multiple digit recognition
   - Mobile optimization
   - REST API endpoint

3. **DevOps Integration**
   - CI/CD pipeline
   - Docker containerization
   - Monitoring dashboard
