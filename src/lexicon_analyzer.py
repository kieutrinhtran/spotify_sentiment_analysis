import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

class LexiconSentimentAnalyzer:
    def __init__(self):
        # Download necessary NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        self.vader = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """Tiền xử lý văn bản"""
        # Chuyển về chữ thường
        text = text.lower()
        # Loại bỏ ký tự đặc biệt và số
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Loại bỏ stopwords
        tokens = [t for t in tokens if t not in self.stop_words]
        return ' '.join(tokens)
    
    def analyze_vader(self, text):
        """Phân tích sentiment sử dụng VADER"""
        scores = self.vader.polarity_scores(text)
        return {
            'compound': scores['compound'],
            'pos': scores['pos'],
            'neu': scores['neu'],
            'neg': scores['neg']
        }
    
    def analyze_textblob(self, text):
        """Phân tích sentiment sử dụng TextBlob"""
        analysis = TextBlob(text)
        return {
            'polarity': analysis.sentiment.polarity,
            'subjectivity': analysis.sentiment.subjectivity
        }
    
    def analyze_reviews(self, reviews_df, text_column='review_text'):
        """Phân tích sentiment cho toàn bộ dataset"""
        results = []
        
        for _, row in reviews_df.iterrows():
            text = row[text_column]
            preprocessed_text = self.preprocess_text(text)
            
            vader_scores = self.analyze_vader(preprocessed_text)
            textblob_scores = self.analyze_textblob(preprocessed_text)
            
            result = {
                'review_id': row.get('review_id', ''),
                'original_text': text,
                'preprocessed_text': preprocessed_text,
                'vader_compound': vader_scores['compound'],
                'vader_pos': vader_scores['pos'],
                'vader_neu': vader_scores['neu'],
                'vader_neg': vader_scores['neg'],
                'textblob_polarity': textblob_scores['polarity'],
                'textblob_subjectivity': textblob_scores['subjectivity']
            }
            results.append(result)
        
        return pd.DataFrame(results)
    
    def get_sentiment_label(self, score, method='vader'):
        """Chuyển đổi điểm sentiment thành nhãn"""
        if method == 'vader':
            if score >= 0.05:
                return 'positive'
            elif score <= -0.05:
                return 'negative'
            else:
                return 'neutral'
        else:  # textblob
            if score > 0:
                return 'positive'
            elif score < 0:
                return 'negative'
            else:
                return 'neutral'
    
    def evaluate_lexicon_methods(self, reviews_df, text_column='review_text'):
        """Đánh giá và so sánh hai phương pháp lexicon-based"""
        results_df = self.analyze_reviews(reviews_df, text_column)
        
        # Thêm nhãn sentiment
        results_df['vader_sentiment'] = results_df['vader_compound'].apply(
            lambda x: self.get_sentiment_label(x, 'vader'))
        results_df['textblob_sentiment'] = results_df['textblob_polarity'].apply(
            lambda x: self.get_sentiment_label(x, 'textblob'))
        
        return results_df 