import pandas as pd

# Đọc file kết quả phân tích lexicon-based và transformer-based
lexicon_df = pd.read_csv('result/lexicon_results.csv')
distilbert_df = pd.read_csv('result/distilbert_results.csv')

# Với lexicon - phân loại sentiment dựa trên tb_label
lexicon_summary = lexicon_df.groupby(['source', 'tb_label']).size().unstack(fill_value=0)

# Với DistilBERT - phân loại sentiment dựa trên dl_label
distilbert_summary = distilbert_df.groupby(['source', 'dl_label']).size().unstack(fill_value=0)

print("Phân bố sentiment theo nền tảng (Lexicon):")
print(lexicon_summary)

print("Phân bố sentiment theo nền tảng (DistilBERT):")
print(distilbert_summary)

lexicon_percent = lexicon_summary.div(lexicon_summary.sum(axis=1), axis=0) * 100
distilbert_percent = distilbert_summary.div(distilbert_summary.sum(axis=1), axis=0) * 100

print("Tỷ lệ phần trăm sentiment theo nền tảng (Lexicon):")
print(lexicon_percent)

print("Tỷ lệ phần trăm sentiment theo nền tảng (DistilBERT):")
print(distilbert_percent)

import matplotlib.pyplot as plt

# Biểu đồ Lexicon
lexicon_percent.plot(kind='bar', stacked=True, figsize=(8,5), title='Phân bố Sentiment theo Nền tảng (Lexicon)')
plt.ylabel('Tỷ lệ phần trăm (%)')
plt.show()

# Biểu đồ DistilBERT
distilbert_percent.plot(kind='bar', stacked=True, figsize=(8,5), title='Phân bố Sentiment theo Nền tảng (DistilBERT)')
plt.ylabel('Tỷ lệ phần trăm (%)')
plt.show()
