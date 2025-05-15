# Bước 2: Viết app vào file app.py (đặt tên file đồng nhất)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Thiết lập trang đầu file
st.set_page_config(
    page_title="📊 Spotify Sentiment Dashboard",
    layout="wide"
)

st.markdown(
    """
    <style>
    h1, h2, h3 {color: #2C3E50;}
    .block-container {padding: 1rem 2rem;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Spotify Sentiment Dashboard")

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

@st.cache_data
def load_data():
    combined = pd.read_csv('result/combined_spotify_reviews.csv')
    lexicon = pd.read_csv('result/lexicon_results.csv')
    distilbert = pd.read_csv('result/distilbert_results.csv')
    combined['date'] = pd.to_datetime(combined['date'], errors='coerce').dt.tz_localize(None)
    return combined, lexicon, distilbert

combined_df, lexicon_df, distilbert_df = load_data()

df = combined_df.rename(columns={'id':'review_id'})
df = df.merge(
    lexicon_df[['review_id','tb_label','vader_compound','tb_prob_pos']],
    on='review_id', how='left'
)
df = df.merge(
    distilbert_df[['review_id','dl_label','confidence']],
    on='review_id', how='left'
)

df['vader_label'] = df['vader_compound'].apply(
    lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
)

df['rating_label'] = df['rating'].apply(
    lambda x: 'negative' if x in [1, 2] else ('neutral' if x == 3 else 'positive')
)

filtered = df.copy()
color_map = {'negative':'#e74c3c', 'neutral':'#95a5a6', 'positive':'#2ecc71'}

tab1, tab2, tab3 = st.tabs([
    "🔍 So sánh phương pháp",
    "📱 So sánh nền tảng",
    "💬 Từ khóa tiêu cực"
])

# Biểu đồ phân bố sentiment
def plot_sentiment_distribution(data, label_col, title):
    """
    Vẽ biểu đồ phân bố sentiment dạng cột chồng.
    
    Args:
        data: DataFrame chứa dữ liệu
        label_col: Tên cột chứa nhãn sentiment
        title: Tiêu đề biểu đồ
    """
    # Tính toán tỷ lệ phần trăm cho mỗi nhãn
    ct = data[label_col].value_counts(normalize=True).sort_index() * 100
    
    # Tạo DataFrame cho biểu đồ
    plot_data = pd.DataFrame({
        'Sentiment': ct.index,
        'Percent': ct.values,
        'Method': [title] * len(ct)
    })
    
    return plot_data

# Hàm tính metric cho lớp positive
def compute_all_metrics(true_labels, pred_labels, labels=None):
    results = {}
    results['Accuracy'] = accuracy_score(true_labels, pred_labels)
    if labels is None:
        labels = sorted(set(true_labels) | set(pred_labels))
    for label in labels:
        results[f'Precision_{label}'] = precision_score(true_labels, pred_labels, labels=[label], average='macro', zero_division=0)
        results[f'Recall_{label}'] = recall_score(true_labels, pred_labels, labels=[label], average='macro', zero_division=0)
        results[f'F1_{label}'] = f1_score(true_labels, pred_labels, labels=[label], average='macro', zero_division=0)
    results['Precision_macro'] = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
    results['Recall_macro'] = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
    results['F1_macro'] = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
    return results

def show_test_and_score_matrix(df, methods, method_names, labels):
    rows = []
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            m_true, m_pred = methods[i], methods[j]
            name_true, name_pred = method_names[i], method_names[j]
            sub_df = df.dropna(subset=[m_true, m_pred])
            metrics = compute_all_metrics(sub_df[m_true], sub_df[m_pred], labels=labels)
            metrics['Phương pháp 1'] = name_true
            metrics['Phương pháp 2'] = name_pred
            rows.append(metrics)

    results_df = pd.DataFrame(rows)
    # Đổi tên cột
    results_df.rename(columns={
        'Accuracy': 'Accuracy',
        'Precision_macro': 'Precision (Macro)',
        'Recall_macro': 'Recall (Macro)',
        'F1_macro': 'F1-score (Macro)',
    }, inplace=True)

    # Định dạng %
    for col in results_df.columns:
        if col not in ['Phương pháp 1', 'Phương pháp 2']:
            results_df[col] = results_df[col].apply(lambda x: f"{x*100:.2f}%")

    st.dataframe(results_df.style.background_gradient(cmap='RdYlGn_r'), use_container_width=True)

def show_method_comparison(df, methods, method_names, labels):
    """
    So sánh và đánh giá độ chính xác giữa các phương pháp phân tích sentiment.
    
    Args:
        df: DataFrame chứa dữ liệu
        methods: List các cột chứa nhãn sentiment
        method_names: List tên các phương pháp
        labels: List các nhãn sentiment
    """
    # Tính toán các chỉ số cho từng phương pháp
    results = []
    for method, name in zip(methods, method_names):
        # Lọc dữ liệu không null
        valid_data = df.dropna(subset=[method])
        
        # Tính toán các chỉ số
        metrics = {
            'Phương pháp': name,
            'Số lượng mẫu': len(valid_data),
            'Tỷ lệ positive': (valid_data[method] == 'positive').mean() * 100,
            'Tỷ lệ negative': (valid_data[method] == 'negative').mean() * 100,
            'Tỷ lệ neutral': (valid_data[method] == 'neutral').mean() * 100
        }
        results.append(metrics)
    
    # Tạo DataFrame kết quả
    results_df = pd.DataFrame(results)
    
    # Định dạng phần trăm
    for col in ['Tỷ lệ positive', 'Tỷ lệ negative', 'Tỷ lệ neutral']:
        results_df[col] = results_df[col].apply(lambda x: f"{x:.1f}%")
    
    # Hiển thị bảng kết quả
    st.dataframe(results_df, use_container_width=True)
    
    # Phân tích và kết luận
    st.subheader("Phân tích kết quả")
    
    # Tính toán các chỉ số so sánh
    distilbert_stats = df[df['dl_label'].notna()]['dl_label'].value_counts(normalize=True) * 100
    lexicon_stats = df[df['tb_label'].notna()]['tb_label'].value_counts(normalize=True) * 100
    
    # Tính độ lệch chuẩn để đánh giá sự cân bằng
    distilbert_std = np.std([distilbert_stats.get('positive', 0), 
                           distilbert_stats.get('negative', 0), 
                           distilbert_stats.get('neutral', 0)])
    lexicon_std = np.std([lexicon_stats.get('positive', 0), 
                         lexicon_stats.get('negative', 0), 
                         lexicon_stats.get('neutral', 0)])
    
    # So sánh tỷ lệ negative
    distilbert_neg = distilbert_stats.get('negative', 0)
    lexicon_neg = lexicon_stats.get('negative', 0)
    
    # Hiển thị kết luận
    st.markdown(f"""
    **Kết luận:**
    1. Phân bố sentiment:
       - DistilBERT: {distilbert_std:.1f}% độ lệch chuẩn
       - Lexicon: {lexicon_std:.1f}% độ lệch chuẩn
       - Phương pháp có phân bố cân bằng hơn: **{"DistilBERT" if distilbert_std < lexicon_std else "Lexicon"}**
    
    2. Tỷ lệ đánh giá tiêu cực:
       - DistilBERT: {distilbert_neg:.1f}%
       - Lexicon: {lexicon_neg:.1f}%
       - Phương pháp phát hiện nhiều đánh giá tiêu cực hơn: **{"DistilBERT" if distilbert_neg > lexicon_neg else "Lexicon"}**
    
    3. Dựa trên phân tích:
       - Phương pháp có độ lệch chuẩn thấp hơn cho thấy phân bố sentiment cân bằng hơn
       - Tỷ lệ đánh giá tiêu cực cao hơn có thể phản ánh thực tế tốt hơn
       - Nên xem xét kết hợp cả hai phương pháp để có đánh giá toàn diện
    """)

# Tab 1
with tab1:
    st.header("1. So sánh sentiment giữa các phương pháp phân tích")
    st.subheader("Phân bố sentiment từng phương pháp")
    
    # Tạo dữ liệu cho biểu đồ
    vader_data = plot_sentiment_distribution(filtered.dropna(subset=['vader_label']), 'vader_label', 'VADER')
    textblob_data = plot_sentiment_distribution(filtered.dropna(subset=['tb_label']), 'tb_label', 'TextBlob')
    distilbert_data = plot_sentiment_distribution(filtered.dropna(subset=['dl_label']), 'dl_label', 'DistilBERT')
    
    # Kết hợp dữ liệu
    combined_data = pd.concat([vader_data, textblob_data, distilbert_data])
    
    # Vẽ biểu đồ cột chồng dọc
    fig = px.bar(
        combined_data,
        x='Method',
        y='Percent',
        color='Sentiment',
        title='Phân bố sentiment theo phương pháp',
        labels={'Method': 'Phương pháp', 'Percent': 'Tỷ lệ (%)'},
        color_discrete_map=color_map,
        template='plotly_dark',
        barmode='stack',
        orientation='v'  # Đặt hướng dọc
    )
    
    # Cập nhật layout
    fig.update_layout(
        height=500,  # Tăng chiều cao để hiển thị rõ hơn
        yaxis=dict(
            ticksuffix='%',
            range=[0, 100]  # Đảm bảo trục y từ 0-100%
        ),
        xaxis=dict(
            tickangle=0  # Giữ nhãn trục x ngang
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        bargap=0.3  # Tăng khoảng cách giữa các cột
    )
    
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Đánh giá độ chính xác của các phương pháp")
    show_method_comparison(filtered, ['vader_label', 'tb_label', 'dl_label'], 
                         ['VADER', 'TextBlob', 'DistilBERT'], 
                         ['positive','neutral','negative'])

# Tab 2
def plot_platform_sentiment(data, label_col, title):
    grp = data.groupby(['source', label_col]).size().unstack(fill_value=0)
    pct = grp.div(grp.sum(axis=1), axis=0) * 100
    chart = pct.reset_index().melt(
        id_vars='source', var_name='Sentiment', value_name='Percent'
    )
    fig = px.bar(
        chart,
        x='source', y='Percent', color='Sentiment', barmode='stack',
        title=title,
        labels={'source':'Nền tảng','Percent':'Tỷ lệ (%)'},
        color_discrete_map=color_map,
        template='plotly_dark'
    )
    fig.update_layout(yaxis=dict(ticksuffix='%'), height=450)
    return fig

with tab2:
    st.header("2. So sánh sentiment giữa các nền tảng")
    method = st.selectbox(
        "Chọn phương pháp phân tích sentiment:",
        ['DistilBERT', 'Lexicon']
    )
    method_map = {
        'DistilBERT': 'dl_label',
        'Lexicon': 'tb_label'
    }
    selected_method = method_map[method]
    method_data = filtered.dropna(subset=[selected_method])

    st.subheader("1. Phân bố sentiment theo nền tảng")
    st.plotly_chart(plot_platform_sentiment(method_data, selected_method, f'{method} Platform'), use_container_width=True)

    st.subheader("2. Thống kê chi tiết theo nền tảng")
    platform_stats = method_data.groupby('source')[selected_method].value_counts(normalize=True).unstack() * 100
    platform_stats = platform_stats.round(2)
    st.dataframe(platform_stats, use_container_width=True)

    st.subheader("3. Phân tích xu hướng sentiment theo thời gian")
    # Chuyển đổi datetime thành date để lấy ngày
    method_data['date'] = pd.to_datetime(method_data['date']).dt.date
    
    # Nhóm dữ liệu theo ngày và sentiment, đếm số lượng review
    time_series = method_data.groupby(['date', selected_method]).size().unstack(fill_value=0)
    time_series = time_series.reset_index()
    
    # Đảm bảo tất cả các cột sentiment có mặt
    for sentiment in ['positive', 'negative', 'neutral']:
        if sentiment not in time_series.columns:
            time_series[sentiment] = 0
    
    # Tạo biểu đồ đường thể hiện số lượng review theo thời gian
    fig_time = px.line(
        time_series,
        x='date',
        y=['positive', 'neutral', 'negative'],
        title=f'Số lượng review theo sentiment theo thời gian - {method}',
        labels={
            'date': 'Ngày',
            'value': 'Số lượng review',
            'variable': 'Sentiment'
        },
        template='plotly_dark',
        height=500
    )
    
    # Cập nhật giao diện biểu đồ
    fig_time.update_layout(
        xaxis_title='Ngày',
        yaxis_title='Số lượng review',
        legend=dict(
            orientation="h",  # Đặt legend nằm ngang
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'  # Hiển thị thông tin khi hover
    )
    
    st.plotly_chart(fig_time, use_container_width=True)

    # Thêm bảng thống kê mô tả số lượng review theo ngày
    st.subheader("4. Thống kê số lượng review")
    # Tính thống kê cho tất cả review theo method đang chọn
    review_stats = method_data[selected_method].value_counts()
    review_stats_df = pd.DataFrame({
        'Sentiment': review_stats.index,
        'Số lượng': review_stats.values,
        'Tỷ lệ (%)': (review_stats.values / len(method_data) * 100).round(2)
    })
    st.dataframe(review_stats_df, use_container_width=True)

# Tab 3
with tab3:
    st.header("3. Phân tích từ khóa tiêu cực")

    col1, col2 = st.columns([1, 2])

    with col1:
        max_words = st.slider("Số lượng từ/cụm từ hiển thị", 10, 100, 20)
        ngram_range = st.slider("Độ dài cụm từ (n-gram)", 1, 3, 2)

    filtered_data = filtered.copy()

    try:
        neg_ids = distilbert_df.loc[distilbert_df['dl_label']=='negative', 'review_id']
        texts_to_plot = filtered_data[filtered_data['review_id'].isin(neg_ids)]['clean_content'].dropna().tolist()

        if not texts_to_plot:
            st.warning('Không có review negative để phân tích')
            st.stop()

        with col2:
            st.subheader("Thống kê cơ bản")
            stats = pd.DataFrame({
                'Chỉ số': ['Số lượng review', 'Tổng số từ', 'Từ trung bình/review'],
                'Giá trị': [
                    len(texts_to_plot),
                    sum(len(t.split()) for t in texts_to_plot),
                    f"{sum(len(t.split()) for t in texts_to_plot)/len(texts_to_plot):.1f}"
                ]
            })
            st.dataframe(stats, use_container_width=True)

        st.subheader("1. WordCloud từ khóa tiêu cực")
        wc = WordCloud(
            width=800, height=300,
            background_color='white',
            colormap='Reds',
            max_words=max_words
        ).generate(' '.join(texts_to_plot))
        fig_wc, ax_wc = plt.subplots(figsize=(10,3))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)

        st.subheader(f"2. Top {max_words} {ngram_range}-gram tiêu cực")
        ngrams_list = []
        for txt in texts_to_plot:
            tokens = [w for w in word_tokenize(txt) if w.isalpha()]
            ngrams_list.extend(list(ngrams(tokens, ngram_range)))
        if not ngrams_list:
            st.warning("Không tìm thấy n-gram nào thỏa mãn điều kiện")
        else:
            top_ngrams = Counter(ngrams_list).most_common(max_words)
            ngram_terms = [' '.join(ng) for ng,_ in top_ngrams]
            ngram_counts = [cnt for _,cnt in top_ngrams]
            fig_ngram = px.bar(
                x=ngram_counts, y=ngram_terms, orientation='h',
                title=f'Top {max_words} {ngram_range}-gram tiêu cực',
                labels={'x':'Tần suất','y':f'{ngram_range}-gram'},
                color=ngram_counts,
                color_continuous_scale='Reds',
                template='plotly_dark'
            )
            fig_ngram.update_layout(yaxis={'autorange':'reversed'}, height=400)
            st.plotly_chart(fig_ngram, use_container_width=True)

        st.subheader("3. Phân tích TF-IDF")
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=max_words,
            min_df=2,
            ngram_range=(1, ngram_range)
        )
        X = vectorizer.fit_transform(texts_to_plot)
        scores = dict(zip(
            vectorizer.get_feature_names_out(),
            X.sum(axis=0).A1
        ))
        if not scores:
            st.warning("Không tìm thấy từ khóa nào thỏa mãn điều kiện TF-IDF")
        else:
            top_tfidf = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_words]
            tf_terms = [term for term,_ in top_tfidf]
            tf_scores = [score for _,score in top_tfidf]
            fig_tf = px.bar(
                x=tf_scores, y=tf_terms, orientation='h',
                title='Top TF-IDF terms',
                labels={'x':'TF-IDF Score','y':'Term'},
                color=tf_scores,
                color_continuous_scale='Reds',
                template='plotly_dark'
            )
            fig_tf.update_layout(yaxis={'autorange':'reversed'}, height=400)
            st.plotly_chart(fig_tf, use_container_width=True)

    except Exception as e:
        st.error(f"Lỗi không xác định: {str(e)}")

# Expander xem dữ liệu thô
with st.expander("📄 Xem dữ liệu thô"):
    st.dataframe(df)  # Show all rows
