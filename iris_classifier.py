import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data():
    """Load and preprocess the Iris dataset"""
    print("Loading Iris dataset...")
    iris = load_iris()
    
    # Convert to DataFrame
    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], 
                     columns=iris['feature_names'] + ['target'])
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    return X, y, df

def split_data(X, y):
    """Split data into training and testing sets"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train decision tree classifier"""
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    predictions = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_test, predictions, average='weighted'
    )
    
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {fscore:.4f}")
    
    return accuracy, precision, recall, fscore

def plot_feature_importance(model, feature_names):
    """Plot feature importance"""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), 
              [feature_names[i] for i in indices], 
              rotation=45)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def main():
    # Load and preprocess data
    X, y, df = load_and_preprocess_data()
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train model
    print("\nTraining Decision Tree Classifier...")
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Plot feature importance
    plot_feature_importance(model, df.columns[:-1])
    print("\nFeature importance plot saved as 'feature_importance.png'")

if __name__ == "__main__":
    main()
