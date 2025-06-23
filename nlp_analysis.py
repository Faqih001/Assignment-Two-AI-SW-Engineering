import spacy
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

def load_spacy_model():
    """Load SpaCy model"""
    print("Loading SpaCy model...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading SpaCy model...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp

def extract_entities(nlp, text):
    """Extract named entities from text"""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PRODUCT']:
            entities.append((ent.text, ent.label_))
    return entities

def analyze_sentiment(text):
    """Analyze sentiment using TextBlob"""
    analysis = TextBlob(text)
    # Get polarity (-1 to 1) and convert to positive/negative
    return 'positive' if analysis.sentiment.polarity > 0 else 'negative'

def process_reviews(reviews):
    """Process a list of reviews"""
    nlp = load_spacy_model()
    results = []
    
    print("\nProcessing reviews...")
    for review in reviews:
        entities = extract_entities(nlp, review)
        sentiment = analyze_sentiment(review)
        results.append({
            'text': review,
            'entities': entities,
            'sentiment': sentiment
        })
    
    return results

def plot_sentiment_distribution(results):
    """Plot sentiment distribution"""
    sentiments = [r['sentiment'] for r in results]
    sentiment_counts = pd.Series(sentiments).value_counts()
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png')
    plt.close()

def main():
    # Sample Amazon product reviews
    sample_reviews = [
        "I love my new Apple iPhone! The camera quality is amazing.",
        "This Samsung TV has terrible picture quality. Not worth the money.",
        "The Sony headphones provide great sound but the build quality is poor.",
        "Nike running shoes are comfortable and durable. Great purchase!",
        "The Amazon Kindle is perfect for reading. Best e-reader ever."
    ]
    
    # Process reviews
    results = process_reviews(sample_reviews)
    
    # Print results
    print("\nAnalysis Results:")
    for i, result in enumerate(results, 1):
        print(f"\nReview {i}:")
        print(f"Text: {result['text']}")
        print(f"Entities: {result['entities']}")
        print(f"Sentiment: {result['sentiment']}")
    
    # Plot sentiment distribution
    plot_sentiment_distribution(results)
    print("\nSentiment distribution plot saved as 'sentiment_distribution.png'")

if __name__ == "__main__":
    main()
