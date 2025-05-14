# Phân tích Sentiment Đánh giá Ứng dụng Spotify

Dự án này thực hiện phân tích sentiment của các đánh giá ứng dụng Spotify trên cả Google Play Store và Apple App Store, sử dụng hai phương pháp chính:

1. **Lexicon-based Approach**:
   - Sử dụng VADER Sentiment Analyzer
   - Sử dụng TextBlob
   - Phân tích dựa trên từ điển cảm xúc

2. **Machine Learning-based Approach**:
   - Sử dụng mô hình BERT
   - Phân tích dựa trên deep learning
   - So sánh kết quả với phương pháp lexicon-based

## Cấu trúc dự án

```
spotify_sentiment_analysis/
├── data/               # Chứa dữ liệu đánh giá
├── src/               # Mã nguồn Python
├── notebooks/         # Jupyter notebooks cho phân tích
└── requirements.txt   # Các thư viện cần thiết
```

## Cài đặt

1. Tạo môi trường ảo:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Cài đặt các thư viện:
```bash
pip install -r requirements.txt
```

## Sử dụng

1. Thu thập dữ liệu đánh giá từ Google Play Store và Apple App Store
2. Chạy phân tích sentiment bằng các phương pháp khác nhau
3. So sánh và đánh giá kết quả

## Kết quả

Dự án sẽ cung cấp:
- Phân tích chi tiết về sentiment của đánh giá
- So sánh giữa hai phương pháp
- Biểu đồ và thống kê
- Báo cáo chi tiết về kết quả phân tích 