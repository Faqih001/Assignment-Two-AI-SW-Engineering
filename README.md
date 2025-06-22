# AI Tools Assignment: Mastering the AI Toolkit 🛠️🧠

This repository contains the implementation of various AI tasks using different frameworks and tools. The assignment is divided into three main parts:

1. Theoretical Understanding
2. Practical Implementation
3. Ethics & Optimization

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
