from transformers import BertTokenizer, BertModel
import torch
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

# BERTのトークナイザーとモデルを読み込む
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 文の例として、テキストを定義
df = pd.read_csv('./nursery_school/nursery_data.csv')
#print(df)
text_list = np.array(df['事業所の理念'])
print(text_list)
#text = "The quick brown fox jumps over the lazy dog."
feature_list = []
feature_color = []

for text in text_list:
	# 文をトークン化し、BERTの入力形式に変換
	input_ids = tokenizer(text, return_tensors='pt', padding=True, truncation=True)['input_ids']
	
	# BERTモデルに入力し、特徴量を取得
	with torch.no_grad():
	    outputs = model(input_ids)
	    features = outputs.last_hidden_state.squeeze(0).numpy()  # BERTの出力から特徴量行列を取得し、NumPy配列に変換
	
	# PCAで次元削減を行う
	pca = PCA(n_components=3)  # 3次元に次元削減
	features_3d = pca.fit_transform(features)
	
	# 変換された特徴量行列の形状を確認
	print("Shape of 3D features:", features_3d.shape)
	feature_list.append(features_3d);
	
	# 各事業所の平均特徴量を計算
	mean_features = np.mean(features_3d, axis=0)
	# 平均特徴量を0から255の範囲にスケーリングして、カラーコードを求める
	min_val = np.min(mean_features)
	max_val = np.max(mean_features)
	scaled_mean_features = (mean_features - min_val) / (max_val - min_val) * 255
	color = '#%02x%02x%02x' % (int(scaled_mean_features[0]), int(scaled_mean_features[1]), int(scaled_mean_features[2]))
	
	# カラーコードを出力
	print("Color for the office:", color)	
	feature_color.append(color)

df['事業所の理念_feature'] = feature_list
df['feature_color'] = feature_color
print(df)
df.to_csv("./nursery_school/nursery_data.csv", index=False)
