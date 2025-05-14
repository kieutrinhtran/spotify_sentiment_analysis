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
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class TransformerSentimentAnalyzer:
    def __init__(self, model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
        """Khởi tạo mô hình Transformer cho phân tích sentiment"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Chọn tokenizer và model phù hợp dựa trên tên model
        if "roberta" in model_name.lower():
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaForSequenceClassification.from_pretrained(model_name)
        elif "distilbert" in model_name.lower():
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            self.model = DistilBertForSequenceClassification.from_pretrained(model_name)
        else:  # Mặc định sử dụng BERT
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForSequenceClassification.from_pretrained(model_name)
            
        self.model.to(self.device)
        self.model_name = model_name
        
    def preprocess_text(self, text, max_length=512):
        """Tiền xử lý văn bản cho mô hình Transformer"""
        return self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    def predict_sentiment(self, text):
        """Dự đoán sentiment cho một văn bản"""
        inputs = self.preprocess_text(text)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        return predictions.cpu().numpy()[0]
    
    def get_sentiment_label(self, prediction):
        """Chuyển đổi kết quả dự đoán thành nhãn sentiment"""
        sentiment_map = {
            0: "very negative",
            1: "negative",
            2: "neutral",
            3: "positive",
            4: "very positive"
        }
        return sentiment_map[np.argmax(prediction)]
    
    def analyze_reviews(self, reviews_df, text_column='review_text'):
        """Phân tích sentiment cho toàn bộ dataset"""
        results = []
        
        for _, row in reviews_df.iterrows():
            text = row[text_column]
            prediction = self.predict_sentiment(text)
            sentiment = self.get_sentiment_label(prediction)
            
            result = {
                'review_id': row.get('review_id', ''),
                'text': text,
                'sentiment': sentiment,
                'confidence': float(np.max(prediction)),
                'prediction_scores': prediction.tolist(),
                'model_name': self.model_name
            }
            results.append(result)
        
        return pd.DataFrame(results)
    
    def evaluate_model(self, reviews_df, text_column='review_text', label_column='sentiment'):
        """Đánh giá hiệu suất của mô hình"""
        # Chia dữ liệu thành tập train và test
        train_df, test_df = train_test_split(reviews_df, test_size=0.2, random_state=42)
        
        # Dự đoán trên tập test
        predictions = []
        true_labels = []
        confidences = []
        
        for _, row in test_df.iterrows():
            text = row[text_column]
            prediction = self.predict_sentiment(text)
            sentiment = self.get_sentiment_label(prediction)
            confidence = float(np.max(prediction))
            
            predictions.append(sentiment)
            true_labels.append(row[label_column])
            confidences.append(confidence)
        
        # Tạo báo cáo phân loại
        report = classification_report(true_labels, predictions, output_dict=True)
        
        # Tạo confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': predictions,
            'true_labels': true_labels,
            'confidences': confidences,
            'model_name': self.model_name
        }
    
    def plot_confusion_matrix(self, confusion_matrix, labels):
        """Vẽ confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def compare_models(self, results_list):
        """So sánh kết quả giữa các mô hình khác nhau"""
        comparison = pd.concat(results_list, axis=0)
        
        # Tính toán các metrics so sánh
        model_metrics = comparison.groupby('model_name').agg({
            'confidence': ['mean', 'std'],
            'sentiment': lambda x: x.value_counts().to_dict()
        })
        
        return model_metrics

# Danh sách các mô hình có thể sử dụng
AVAILABLE_MODELS = {
    'bert': 'nlptown/bert-base-multilingual-uncased-sentiment',
    'roberta': 'cardiffnlp/twitter-roberta-base-sentiment',
    'distilbert': 'distilbert-base-uncased-finetuned-sst-2-english'
} 