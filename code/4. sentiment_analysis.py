# Phân tích Sentiment Đánh giá Ứng dụng Spotify
# Sử dụng cả phương pháp Lexicon-based và Deep Learning-based
# Lexicon-based: VADER và TextBlob
# Deep Learning-based: DistilBERT

import sys
sys.path.append('..')

# Thư viện xử lý dữ liệu và trực quan hóa
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Thư viện Deep Learning - sử dụng cho mô hình transformer
import torch
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification
)

# Thư viện phân tích sentiment - sử dụng cho phương pháp lexicon-based
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

# Thư viện xử lý ngôn ngữ tự nhiên - dùng để tiền xử lý văn bản
import nltk
from nltk.corpus import stopwords

# Thư viện đánh giá mô hình - dùng để đánh giá kết quả phân loại
from sklearn.metrics import classification_report

# Thư viện logging
import logging
from pathlib import Path
from datetime import datetime

# Tạo thư mục logs nếu chưa tồn tại
Path('logs').mkdir(exist_ok=True)
Path('result').mkdir(exist_ok=True)

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/sentiment_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

# Tải các dữ liệu cần thiết cho NLTK
nltk.download('punkt')  # Dùng để tokenize văn bản
nltk.download('stopwords')  # Danh sách stopwords tiếng Anh
nltk.download('vader_lexicon')  # Lexicon cho VADER sentiment
nltk.download('movie_reviews')  # Corpus cần thiết cho TextBlob NaiveBayesAnalyzer

# Định nghĩa mô hình transformer có sẵn để sử dụng
AVAILABLE_MODELS = {
    'distilbert': 'distilbert-base-uncased-finetuned-sst-2-english'  # DistilBERT tiếng Anh, phân loại positive/negative
}

class LexiconSentimentAnalyzer:
    """
    Lớp phân tích sentiment sử dụng phương pháp lexicon-based
    Kết hợp cả VADER và TextBlob để có kết quả đa dạng
    """
    def __init__(self, batch_size=32):
        self.vader = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        self.batch_size = batch_size

    def analyze_vader_batch(self, texts):
        """
        Phân tích sentiment sử dụng VADER cho một batch văn bản
        """
        results = []
        for text in texts:
            try:
                vs = self.vader.polarity_scores(str(text))
                results.append({
                    'vader_compound': vs['compound'],
                    'vader_pos': vs['pos'],
                    'vader_neu': vs['neu'],
                    'vader_neg': vs['neg']
                })
            except Exception as e:
                logging.warning(f"VADER analysis failed for text: {str(e)}")
                results.append({
                    'vader_compound': 0.0,
                    'vader_pos': 0.0,
                    'vader_neu': 1.0,
                    'vader_neg': 0.0
                })
        return pd.DataFrame(results)

    def analyze_textblob_batch(self, texts):
        """
        Phân tích sentiment sử dụng TextBlob cho một batch văn bản
        """
        results = []
        label_map = {
            'pos': 'positive',
            'neg': 'negative',
            'neu': 'neutral'
        }
        for text in texts:
            try:
                blob = TextBlob(str(text), analyzer=NaiveBayesAnalyzer())
                sent = blob.sentiment
                results.append({
                    'tb_label': label_map.get(sent.classification, 'neutral'),
                    'tb_prob_pos': sent.p_pos,
                    'tb_prob_neg': sent.p_neg
                })
            except Exception as e:
                logging.warning(f"TextBlob analysis failed for text: {str(e)}")
                results.append({
                    'tb_label': 'neutral',
                    'tb_prob_pos': 0.5,
                    'tb_prob_neg': 0.5
                })
        return pd.DataFrame(results)

    def analyze_reviews(self, df, text_column='clean_content'):
        """
        Phân tích sentiment cho toàn bộ dataset sử dụng batch processing
        """
        texts = df[text_column].astype(str).tolist()
        n_samples = len(texts)
        
        # Khởi tạo DataFrame kết quả với cùng index như df đầu vào và đúng dtypes
        results_df = pd.DataFrame(index=df.index)
        results_df['review_id'] = df['id']
        results_df['source'] = df['source']
        results_df['vader_compound'] = 0.0
        results_df['vader_pos'] = 0.0
        results_df['vader_neu'] = 0.0
        results_df['vader_neg'] = 0.0
        results_df['tb_label'] = ''  # Initialize as string
        results_df['tb_prob_pos'] = 0.0
        results_df['tb_prob_neg'] = 0.0
        
        # Tạo progress bar
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        pbar = tqdm(total=n_batches, desc='Lexicon Analysis Progress', unit='batch')

        # Process in batches
        for i in range(0, n_samples, self.batch_size):
            batch_start_time = time.time()
            batch_texts = texts[i:i + self.batch_size]
            batch_indices = df.index[i:i + len(batch_texts)]
            
            # Phân tích VADER cho batch
            vader_results = self.analyze_vader_batch(batch_texts)
            vader_results.index = batch_indices
            
            # Phân tích TextBlob cho batch
            textblob_results = self.analyze_textblob_batch(batch_texts)
            textblob_results.index = batch_indices
            
            # Cập nhật kết quả vào DataFrame chính
            results_df.loc[batch_indices, vader_results.columns] = vader_results
            results_df.loc[batch_indices, textblob_results.columns] = textblob_results
            
            # Cập nhật progress bar
            batch_time = time.time() - batch_start_time
            pbar.set_postfix({'batch_time': f'{batch_time:.2f}s'})
            pbar.update(1)

        pbar.close()
        
        # Đảm bảo thứ tự cột giống như ban đầu
        results_df = results_df[[
            'review_id', 'source', 'vader_compound', 'vader_pos', 
            'vader_neu', 'vader_neg', 'tb_label', 'tb_prob_pos', 'tb_prob_neg'
        ]]
        
        return results_df

class TransformerSentimentAnalyzer:
    """
    Lớp phân tích sentiment sử dụng mô hình transformer (DistilBERT)
    Xử lý phân loại positive/negative
    """
    def __init__(self, model_name, batch_size=32):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode
        self.batch_size = batch_size
        self.model_name = model_name

    def preprocess_batch(self, texts, max_length=512):
        """
        Tiền xử lý một batch văn bản
        """
        return self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

    def predict_batch(self, texts):
        """
        Dự đoán sentiment cho một batch văn bản
        """
        inputs = self.preprocess_batch(texts)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        return probs.cpu().numpy()

    def get_labels(self, probs):
        """
        Chuyển đổi xác suất thành nhãn sentiment cho một batch
        """
        return ['positive' if np.argmax(p)==1 else 'negative' for p in probs]

    def analyze_reviews(self, df, text_column='clean_content'):
        """
        Phân tích sentiment cho toàn bộ dataset sử dụng batch processing
        """
        texts = df[text_column].astype(str).tolist()
        n_samples = len(texts)
        results = []

        # Tạo progress bar
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        pbar = tqdm(total=n_batches, desc='Transformer Analysis Progress', unit='batch')

        # Process in batches
        for i in range(0, n_samples, self.batch_size):
            batch_start_time = time.time()
            batch_texts = texts[i:i + self.batch_size]
            probs = self.predict_batch(batch_texts)
            labels = self.get_labels(probs)
            confidences = np.max(probs, axis=1)

            # Add results for this batch
            for j, (label, conf) in enumerate(zip(labels, confidences)):
                idx = i + j
                if idx < n_samples:
                    results.append({
                        'review_id': df.iloc[idx].get('id', ''),
                        'source': df.iloc[idx].get('source', ''),
                        'dl_label': label,
                        'confidence': float(conf)
                    })

            # Cập nhật progress bar
            batch_time = time.time() - batch_start_time
            pbar.set_postfix({
                'batch_time': f'{batch_time:.2f}s',
                'device': str(self.device)
            })
            pbar.update(1)

        pbar.close()
        return pd.DataFrame(results)

if __name__ == '__main__':
    try:
        start_time = time.time()
        
        # 1. Load dữ liệu từ file CSV
        logging.info('Loading data from CSV...')
        df = pd.read_csv('data/combined_spotify_reviews.csv',
                        parse_dates=['date','scraped_at'])
        df['clean_content'] = df['clean_content'].fillna('')
        logging.info(f'Loaded {len(df)} reviews')

        # 2. Phân tích sử dụng phương pháp lexicon-based với batch size 32
        logging.info('Starting lexicon-based analysis...')
        lex = LexiconSentimentAnalyzer(batch_size=32)
        lex_df = lex.analyze_reviews(df)
        lex_df.to_csv('result/lexicon_results.csv', index=False)
        logging.info('Lexicon-based analysis completed')

        # 3. Phân tích sử dụng mô hình transformer
        logging.info('Starting DistilBERT analysis...')
        dl = TransformerSentimentAnalyzer(AVAILABLE_MODELS['distilbert'], batch_size=32)
        out_df = dl.analyze_reviews(df)
        out_df.to_csv('result/distilbert_results.csv', index=False)
        logging.info('DistilBERT analysis completed')

        total_time = time.time() - start_time
        logging.info(f'All analyses completed successfully in {total_time:.2f} seconds')

    except Exception as e:
        logging.error(f'An error occurred: {str(e)}', exc_info=True)
        raise