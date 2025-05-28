# Spotify Sentiment Analysis

Dự án phân tích cảm xúc (sentiment analysis) của người dùng đối với ứng dụng Spotify trên cả hai nền tảng Google Play Store và Apple App Store.
![image](https://github.com/user-attachments/assets/1b5ba0cc-c499-4d51-aff6-ae5494d93621)

---

## Clone repository

Để bắt đầu, bạn cần clone mã nguồn từ GitHub về máy:

```bash
git clone https://github.com/kieutrinhtran/spotify_sentiment_analysis.git
cd spotify_sentiment_analysis
```
## Cấu trúc dự án

```
spotify_sentiment_analysis/
├── code/                           # Thư mục chứa mã nguồn
│   ├── 0. spotify_reviews_scraper_ggplaystore.py    # Thu thập đánh giá từ Google Play Store
│   ├── 1. spotify_reviews_scraper_applestore.py     # Thu thập đánh giá từ Apple App Store
│   ├── 2. pre_process.py           # Tiền xử lý dữ liệu
│   ├── 3. HDFS_storage.py          # Lưu trữ dữ liệu vào HDFS
│   ├── 4. sentiment_analysis.py    # Phân tích cảm xúc
│   ├── 5. sentiment_comparison_two_platforms.py     # So sánh cảm xúc giữa hai nền tảng
│   ├── 6. extract_negative_words.py # Trích xuất từ ngữ tiêu cực
│   ├── 7. run_with_ngrok.py        # Chạy ứng dụng với ngrok
│   └── app.py                      # Ứng dụng Streamlit chính
├── data/                           # Thư mục chứa dữ liệu gốc
├── data-test/                      # Thư mục chứa dữ liệu test
├── result/                         # Thư mục chứa kết quả phân tích
├── logs/                           # Thư mục chứa log
├── requirements.txt                # Danh sách các thư viện cần thiết
└── README.md                       # Tài liệu hướng dẫn
```

## Cài đặt

1. Tạo môi trường ảo Python:
```bash
python -m venv .venv
```

2. Kích hoạt môi trường ảo:
- Windows:
```bash
.venv\Scripts\activate
```
- Linux/Mac:
```bash
source .venv/bin/activate
```

3. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## Các tính năng chính

1. **Thu thập dữ liệu**:
   - Thu thập đánh giá từ Google Play Store
   - Thu thập đánh giá từ Apple App Store

2. **Tiền xử lý dữ liệu**:
   - Làm sạch văn bản
   - Xử lý emoji và ký tự đặc biệt
   - Chuẩn hóa dữ liệu

3. **Phân tích cảm xúc**:
   - Sử dụng nhiều mô hình phân tích cảm xúc khác nhau
   - So sánh kết quả giữa các mô hình
   - Trích xuất từ ngữ tiêu cực

4. **Trực quan hóa dữ liệu**:
   - Biểu đồ phân bố cảm xúc
   - Word cloud cho các từ ngữ tiêu cực
   - So sánh cảm xúc giữa hai nền tảng

## Chạy ứng dụng

1. Chạy ứng dụng Streamlit:
```bash
streamlit run code/app.py
```

2. Chạy với ngrok (để chia sẻ ứng dụng):
```bash
python code/run_with_ngrok.py
```

## Các thư viện chính

- pandas, numpy: Xử lý dữ liệu
- scikit-learn: Machine learning
- nltk, textblob, vaderSentiment: Phân tích cảm xúc
- transformers, torch: Deep learning
- streamlit: Giao diện web
- plotly: Trực quan hóa dữ liệu
- wordcloud: Tạo word cloud
- google-play-scraper: Thu thập dữ liệu từ Google Play Store
