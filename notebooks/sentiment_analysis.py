# Phân tích Sentiment Đánh giá Ứng dụng Spotify
# Sử dụng cả phương pháp Lexicon-based và Machine Learning-based
# Lexicon-based: VADER và TextBlob
# Machine Learning-based: BERT, RoBERTa, DistilBERT

import sys
sys.path.append('..')

# Thư viện xử lý dữ liệu và trực quan hóa
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Thư viện Deep Learning
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertTokenizer,
    BertForSequenceClassification,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)

# Thư viện phân tích sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

# Thư viện xử lý ngôn ngữ tự nhiên
import nltk
from nltk.corpus import stopwords

# Thư viện đánh giá mô hình
from sklearn.metrics import classification_report

# Tải dữ liệu cần thiết cho NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Danh sách các mô hình transformer
AVAILABLE_MODELS = {
    'bert': 'nlptown/bert-base-multilingual-uncased-sentiment',
    'roberta': 'cardiffnlp/twitter-roberta-base-sentiment',
    'distilbert': 'distilbert-base-uncased-finetuned-sst-2-english'
}

class LexiconSentimentAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))

    def analyze_vader(self, text):
        scores = self.vader.polarity_scores(text)
        return scores

    def analyze_textblob(self, text):
        # Sử dụng NaiveBayesAnalyzer để tránh thiếu pattern
        blob = TextBlob(text, analyzer=NaiveBayesAnalyzer())
        sentiment = blob.sentiment
        return {
            'polarity': sentiment.classification,  # positive/negative
            'prob_positive': sentiment.p_pos,
            'prob_negative': sentiment.p_neg
        }

    def analyze_reviews(self, reviews_df, text_column='clean_content'):
        results = []
        for _, row in reviews_df.iterrows():
            text = row[text_column]
            vader_scores = self.analyze_vader(text)
            tb_scores = self.analyze_textblob(text)
            result = {
                'review_id': row.get('id', ''),
                'source': row.get('source', ''),
                'text': text,
                'vader_compound': vader_scores['compound'],
                'vader_pos': vader_scores['pos'],
                'vader_neu': vader_scores['neu'],
                'vader_neg': vader_scores['neg'],
                'tb_label': tb_scores['polarity'],
                'tb_prob_pos': tb_scores['prob_positive'],
                'tb_prob_neg': tb_scores['prob_negative']
            }
            results.append(result)
        return pd.DataFrame(results)

class TransformerSentimentAnalyzer:
    def __init__(self, model_name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if 'roberta' in model_name:
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaForSequenceClassification.from_pretrained(model_name)
        elif 'distilbert' in model_name:
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            self.model = DistilBertForSequenceClassification.from_pretrained(model_name)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model_name = model_name

    def preprocess(self, text, max_length=512):
        return self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

    def predict(self, text):
        inputs = self.preprocess(text)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model(**inputs)
            scores = torch.nn.functional.softmax(output.logits, dim=-1)
        return scores.cpu().numpy()[0]

    def get_label(self, scores):
        # Áp dụng mapping phù hợp cho model nlptown
        idx = np.argmax(scores)
        if 'multilingual' in self.model_name:
            # idx 0->1 sao, 4->5 sao
            return f'{idx+1}_stars'
        labels = ['negative', 'neutral', 'positive']
        return labels[idx]

    def analyze_reviews(self, reviews_df, text_column='clean_content'):
        results = []
        for _, row in reviews_df.iterrows():
            text = row[text_column]
            scores = self.predict(text)
            label = self.get_label(scores)
            results.append({
                'review_id': row.get('id', ''),
                'source': row.get('source', ''),
                'text': text,
                'label': label,
                'confidence': float(np.max(scores))
            })
        return pd.DataFrame(results)

# Main execution
if __name__ == '__main__':
    # Load data
    df = pd.read_csv('../data/combined_spotify_reviews.csv', parse_dates=['date','scraped_at'])
    print(df.info())

    # Lexicon-based
    lex = LexiconSentimentAnalyzer()
    lex_results = lex.analyze_reviews(df)
    lex_results.to_csv('../results/lexicon_results.csv', index=False)

    # Transformer-based
    transformer_results = {}
    for name, path in AVAILABLE_MODELS.items():
        analyzer = TransformerSentimentAnalyzer(path)
        transformer_results[name] = analyzer.analyze_reviews(df)
        transformer_results[name].to_csv(f'../results/{name}_results.csv', index=False)

    print('Analysis completed.')
