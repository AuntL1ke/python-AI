import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from textblob import TextBlob
import re


# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()


with open("reviews.json", "r", encoding="utf-8") as f:
    data = json.load(f)

sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}

for review in data:
    text = review["Comment Text"]
    text_clean = re.sub(r"[^\w\s]", "", text.lower())
    tokens = nltk.word_tokenize(text_clean)
    filtered = [word for word in tokens if word not in stop_words]
    stemmed = [stemmer.stem(word) for word in filtered]
    lemmatized = [lemmatizer.lemmatize(word) for word in stemmed]


    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.1:
        sentiment = "positive"
    elif polarity < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    sentiment_counts[sentiment] += 1

  
    print("-" * 60)
    print(f"Author: {review['Author']}")
    # print(f"Comment: {text[:120]}{'...' if len(text) > 120 else ''}")
    print(f"Polarity: {polarity:.2f}")
    print(f"Sentiment: {sentiment}")

print("\nSentiment summary:")
print(json.dumps(sentiment_counts, indent=2))
