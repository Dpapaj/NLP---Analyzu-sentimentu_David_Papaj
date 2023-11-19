import requests
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Get reviews from IMDB page
"""
url = 'https://www.imdb.com/title/tt1853728/reviews'   
url = 'https://www.imdb.com/title/tt0109830/reviews'
url = 'https://www.imdb.com/title/tt0903747/reviews'
url = 'https://www.imdb.com/title/tt5734576/reviews'
"""
url = 'https://www.imdb.com/title/tt1213644/reviews'

response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
reviews = soup.find_all('div', {'class': 'text show-more__control'})

# Perform sentiment analysis on reviews
sia = SentimentIntensityAnalyzer()
positive = 0
negative = 0
neutral = 0
sentiments = []
for review in reviews:
    text = review.get_text()
    score = sia.polarity_scores(text)['compound']
    if score > 0.05:
        positive += 1
        sentiments.append('Positive')
    elif score < -0.05:
        negative += 1
        sentiments.append('Negative')
    else:
        neutral += 1
        sentiments.append('Neutral')

sentiment_counts = Counter(sentiments)
wordcloud = WordCloud(background_color='white', width=1200, height=800).generate_from_frequencies(sentiment_counts)
plt.figure(figsize=(12, 8))
plt.suptitle('IMDB Reviews Analysis\n{}'.format(url), fontsize=16)
plt.subplot(2, 3, 1)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Sentiment Analysis Results\n positive {}, negative {}, neutral {}\n{} Reviews'.format(positive, negative, neutral, len(reviews)))


# Get all words from reviews
words = []
for review in reviews:
    text = review.get_text()
    text = text.lower()
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])
    words += text.split()

# Remove stop words from words list
stop_words = set(stopwords.words('english'))
words = [word for word in words if word not in stop_words]

# Get 30 most common words
word_counts = Counter(words)
top_words = word_counts.most_common(30)

# Generate word cloud of 30 most common words
wordcloud = WordCloud(background_color='white', width=1200, height=800).generate_from_frequencies(dict(top_words))
plt.subplot(2, 3, 2)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('30 Most Used Words')

# Get 30 longest words
longest_words = sorted(words, key=lambda word: len(word), reverse=True)[:30]

# Generate word cloud of 30 longest words
wordcloud = WordCloud(background_color='white', width=1200, height=800).generate(' '.join(longest_words))
plt.subplot(2, 3, 3)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('30 Longest Words')

# Display sentiment analysis results
labels = ['Positive', 'Negative', 'Neutral']
sizes = [positive, negative, neutral]
colors = ['#33cc33', '#ff3300', '#ffff33']
plt.subplot(2, 3, 4)
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Sentiment Analysis Results ')

# Display bar chart of 30 most used words
words = [word[0] for word in top_words]
counts = [word[1] for word in top_words]
plt.subplot(2, 3, 5)
plt.bar(words, counts, color='#0099cc')
plt.xticks(rotation=90)
plt.title('30 Most Used Words')

# Display bar chart of 30 longest words
longest_words = sorted(words, key=len, reverse=True)[:30]
lengths = [len(word) for word in longest_words]
plt.subplot(2, 3, 6)
plt.bar(longest_words, lengths, color='#0099cc')
plt.xticks(rotation=90)
plt.title('30 Longest Words')

plt.tight_layout()
plt.show()
