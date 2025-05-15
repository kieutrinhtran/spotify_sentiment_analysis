import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer

# 1. Đọc kết quả từ hai phương pháp
lex_df = pd.read_csv('result/lexicon_results.csv')
dl_df = pd.read_csv('result/distilbert_results.csv')

# 2. Ghép hai bảng theo review_id
merged_df = pd.merge(lex_df, dl_df, on='review_id', how='inner')

# 3. Chuẩn hóa nhãn TextBlob thành lex_label
merged_df['lex_label'] = merged_df['tb_label'].map({
    'pos': 'positive',
    'neg': 'negative'
}).fillna('neutral')

# 4. So sánh nhãn lexicon vs deep learning
merged_df['match'] = merged_df['lex_label'] == merged_df['dl_label']
agreement_rate = merged_df['match'].mean()
print(f"Tỷ lệ trùng khớp giữa hai phương pháp: {agreement_rate:.2%}")

# 5. Phân tích sự khác biệt cảm xúc giữa hai nền tảng
sentiment_source_df = merged_df.groupby(['source', 'dl_label']).size().unstack(fill_value=0)
print("\nPhân bố sentiment theo nền tảng:")
print(sentiment_source_df)

# Vẽ biểu đồ phân bố sentiment
plt.figure(figsize=(8, 5))
sentiment_source_df.plot(kind='bar', stacked=True)
plt.title('Phân bố sentiment theo nguồn đánh giá')
plt.xlabel('Nguồn (source)')
plt.ylabel('Số lượng đánh giá')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 6. Trích xuất từ khóa tiêu cực (chỉ dùng nhãn từ deep learning)
negative_df = merged_df[merged_df['dl_label'] == 'negative']
vectorizer = CountVectorizer(stop_words='english', max_features=20)
X = vectorizer.fit_transform(negative_df['clean_content'].fillna(''))

# Tạo bảng từ khóa
keywords = pd.DataFrame({
    'keyword': vectorizer.get_feature_names_out(),
    'frequency': X.sum(axis=0).A1
}).sort_values(by='frequency', ascending=False)

print("\nTop 20 từ khóa tiêu cực phổ biến:")
print(keywords)

# (Tùy chọn) Vẽ biểu đồ từ khóa tiêu cực
plt.figure(figsize=(10, 5))
sns.barplot(data=keywords, x='frequency', y='keyword', palette='Reds_r')
plt.title('Top 20 từ khóa phổ biến trong đánh giá tiêu cực (DistilBERT)')
plt.xlabel('Tần suất xuất hiện')
plt.ylabel('Từ khóa')
plt.tight_layout()
plt.show()
