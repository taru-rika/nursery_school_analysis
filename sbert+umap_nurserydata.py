from transformers import BertJapaneseTokenizer, BertModel
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import japanize_matplotlib
import umap.umap_ as umap
import hdbscan
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import shap
from sklearn.neighbors import KNeighborsClassifier

# BERTのトークナイザーとモデルを読み込む
tokenizer = BertJapaneseTokenizer.from_pretrained('sonoisa/sentence-bert-base-ja-mean-tokens-v2')
model_bert = BertModel.from_pretrained('sonoisa/sentence-bert-base-ja-mean-tokens-v2')
model_bert.eval()

# CSVファイルを読み込む
df = pd.read_csv('./nursery_school/nursery_data2.csv')
#df = pd.read_csv('./nursery_school/unlicensed_nursery_data.csv' ,skip_blank_lines=True)
df.index = range(len(df))
df = df.dropna(how='all')  # すべての列がNaNである行を削除
df = df.dropna(subset=['analyze_service'])
#text_list = np.array(df['方針・理念'].iloc[1:].astype(str) + ' ' + df['teacher'].iloc[1:].astype(str) + ' ' + df['保育・教育内容'].iloc[1:].astype(str))
text_list = np.array(df['analyze_service'])
specific_string = "共" + "通" + "評" + "価" + "項" + "目" + "\\" + "n" + "\\" + "t" + "\\" + "t" + "\\" + "t" + "\\" + "t"
text_new_list = []
for text in text_list:
	text = text.replace('[','')
	text = text.replace(']','')
	text = text.replace(',','')
	start_index = text.find(specific_string)
	text = text[start_index + len(specific_string):]
	print(text)
	text_new_list.append(text)
	
#text_list = str(text_list)
print(text_new_list)
print(len(text_new_list))

# 'facilityName'の列を取得
facility_names = df['facilityName'].iloc[1:].astype(str).values

# テキストをBERTでベクトルに変換する関数
def get_sentence_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
    with torch.no_grad():
        outputs = model_bert(**inputs)
    # [CLS]トークンのベクトルを使用 (最初のトークン)
    return outputs.last_hidden_state[:, 0, :].numpy()
# 全テキストを768次元ベクトルに変換
embeddings = np.array([get_sentence_embedding(text) for text in text_new_list])
embeddings = embeddings.reshape(len(text_new_list), -1)

# 標準化（UMAPに適用する前に推奨されるステップ）
scaler = StandardScaler()
scaled_embeddings = scaler.fit_transform(embeddings)

### 一旦2次元に圧縮したものを可視化してみる
## UMAPで2次元に圧縮
#umap_model = umap.UMAP(n_components=2, random_state=42)
#reduced_embeddings = umap_model.fit_transform(scaled_embeddings)
#
## 2次元ベクトルをプロット
#plt.figure(figsize=(10, 10))
#plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], marker='o')
#
## グラフにラベルを付ける（任意）
#for i, facility_name in enumerate(facility_names):
#    plt.annotate(facility_name, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
#
#plt.title('UMAPによるBERT埋め込みの2次元可視化')
#plt.xlabel('UMAP 1')
#plt.ylabel('UMAP 2')
#plt.grid(True)
#plt.show()

# UMAPで768次元を10次元に圧縮
umap_20d = umap.UMAP(n_components=20, random_state=42, n_jobs=1)
reduced_embeddings_20d = umap_20d.fit_transform(scaled_embeddings)
# PCAで768次元を20次元に削減
#pca_20d = PCA(n_components=20)
#reduced_embeddings_20d = pca_20d.fit_transform(scaled_embeddings)

# HDBSCANによるクラスタリング
#clusterer = hdbscan.HDBSCAN(min_cluster_size=3, gen_min_span_tree=True)
# k-meansによるクラスタリング
clusterer = KMeans(n_clusters=5, random_state=42)
cluster_labels = clusterer.fit_predict(reduced_embeddings_20d)
# クラスタリング結果を新しい列に保存
df['cluster'] = cluster_labels

## SHAP値による可視化
# クラスタリング結果を近似するためのKNN分類器を作成
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(reduced_embeddings_20d, cluster_labels)

# SHAPのExplainerを作成し、SHAP値を計算
explainer = shap.KernelExplainer(knn.predict_proba, reduced_embeddings_20d)
shap_values = explainer.shap_values(reduced_embeddings_20d)

# SHAP値を可視化
shap.summary_plot(shap_values, features=reduced_embeddings_20d, feature_names=["UMAP1", "UMAP2"], plot_type="bar")

## UMAPによる2次元可視化
# クラスタリング結果を2次元に圧縮して可視化
umap_2d = umap.UMAP(n_components=2, random_state=42)
reduced_embeddings_2d = umap_2d.fit_transform(reduced_embeddings_20d)

# クラスタリング結果をプロット
plt.figure(figsize=(10, 10))
plt.scatter(reduced_embeddings_2d[:, 0], reduced_embeddings_2d[:, 1], c=cluster_labels, cmap='Spectral', s=50)
plt.colorbar(boundaries=np.arange(max(cluster_labels) + 2) - 0.5).set_ticks(np.arange(max(cluster_labels) + 1))

# 各クラスタの点に施設名のラベルを付ける（任意）
for i, facility_name in enumerate(facility_names):
    plt.annotate(facility_name, (reduced_embeddings_2d[i, 0], reduced_embeddings_2d[i, 1]))

#plt.title('UMAP(20次元) & HDBSCANによるクラスタリング結果の2次元プロット')
#plt.title('PCA(20次元) & HDBSCANによるクラスタリング結果の2次元プロット')
plt.title('UMAP(20次元) & k-meansによるクラスタリング結果の2次元プロット')
#plt.title('PCA(20次元) & k-meansによるクラスタリング結果の2次元プロット')

plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.grid(True)
plt.show()

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-no_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]
        
        # コンソールには単語のみ表示
        print(f"Topic {topic_idx}: {' '.join(top_features)}")

        fig, ax = plt.subplots()
        ax.barh(top_features, weights, color='blue')
        ax.set_xlabel('Importance')
        ax.set_title('Top 5 words for Topic {}'.format(topic_idx))
        plt.gca().invert_yaxis()
        plt.show()

# クラスタごとのテキストデータを抽出
cluster_texts = {i: [] for i in np.unique(cluster_labels)}
for text, label in zip(text_new_list, cluster_labels):
    cluster_texts[label].append(text)

# 各クラスタについてトピックモデリングを行う
no_top_words = 5  # 各トピックから上位5語を抽出
for cluster, texts in cluster_texts.items():
    print(f"Cluster {cluster} Topics:")
    if texts:  # テキストがある場合のみ処理
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
        tfidf = tfidf_vectorizer.fit_transform(texts)
        nmf = NMF(n_components=5, random_state=1).fit(tfidf)
        tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
        display_topics(nmf, tfidf_feature_names, no_top_words)
    else:
        print("No data available.")

# クラスタラベルに対応する色を決める関数
def assign_colors_to_clusters(cluster_labels):
    unique_labels = np.unique(cluster_labels)
    cmap = plt.get_cmap("Spectral", len(unique_labels))  # 色のマップを作成
    color_mapping = {label: matplotlib.colors.rgb2hex(cmap(i)[:3]) for i, label in enumerate(unique_labels)}  # クラスタラベルごとに色を割り当て
    return [color_mapping[label] for label in cluster_labels]

# クラスタラベルに基づいて色を割り当て
df['feature_color'] = assign_colors_to_clusters(cluster_labels)

# 更新されたデータをCSVファイルに保存
df.to_csv('./nursery_school/nursery_data2_update.csv', index=False)

# CSVファイル保存完了のメッセージを表示
print("クラスタリング結果に基づく 'feature_color' 列が追加され、CSVファイルに保存されました。")