from transformers import BertJapaneseTokenizer, BertModel
#from transformers import BertTokenizer, BertModel
import torch
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import base64
from io import BytesIO
from wordcloud import WordCloud
import colorsys

# BERTのトークナイザーとモデルを読み込む
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
model_bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
model_bert.eval()

# 文の例として、テキストを定義
df = pd.read_csv('./nursery_school/nursery_data.csv')
#print(df)
text_list = np.array(df['事業所の理念'])
print(text_list)
#text = "The quick brown fox jumps over the lazy dog."
feature_color = []
feature_list = []
img_list = []

def calc_embedding(text):
  bert_tokens = tokenizer.tokenize(text)
  ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + bert_tokens[:126] + ["[SEP]"])
  tokens_tensor = torch.tensor(ids).reshape(1, -1)

  with torch.no_grad():
    output = model_bert(tokens_tensor)

  #return output[1].numpy()
  return output.last_hidden_state.squeeze(0).numpy()

def float_to_rgb(float_values):
    # 各float値が[0, 1]の範囲であることを確認
    assert all(0.0 <= val <= 1.0 for val in float_values), "Float values should be in the range [0, 1]"
    # 平均値を計算
    mean_value = sum(float_values) / len(float_values)
    # 平均値に基づいてスケーリングを調整
    adjusted_values = [(val - mean_value) * 0.5 + 0.5 for val in float_values]
    # HSLの色相、彩度、輝度を計算（ここでは彩度と輝度を固定値にしています）
    hue = adjusted_values[0]  # 色相
    saturation = adjusted_values[1]  # 彩度
    lightness = adjusted_values[2]  # 輝度
    # HSLからRGBへの変換
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    # [0, 255]の範囲にスケーリングし、整数に変換
    rgb_values = [int(r * 255), int(g * 255), int(b * 255)]
    return rgb_values

for text in text_list:
	# 特徴量行列を取得
	feature = calc_embedding(text)
	# PCAで次元削減を行う
	pca = PCA(n_components=3)  # 3次元に次元削減
	features_3d = pca.fit_transform(feature)
	result_r = pca.explained_variance_ratio_[0]
	result_g = pca.explained_variance_ratio_[1]
	result_b = pca.explained_variance_ratio_[2]
	rgb_list = [result_r, result_g, result_b]
	print(rgb_list)
	rgb_values = float_to_rgb(rgb_list)
	color = '#%02x%02x%02x' % (int(rgb_values[0]), int(rgb_values[1]), int(rgb_values[2]))
	#print(color)
	feature_color.append(color)

	# wordcloudのエンコード情報を取得
	#フォントの設定
	#font_path="/usr/share/fonts/truetype/fonts-japanese-gothic.ttf"
	#font_path='/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc'
	font_path='/Users/rikatarumi/Library/Fonts/NotoSansJP-VariableFont_wght.ttf'
	#色の設定
	colormap="Paired"
	# WordCloudの生成（日本語フォントを指定）
	wordcloud = WordCloud(font_path=font_path, width=400, height=200, background_color='white', stopwords=["1", "2", "3", "4"]).generate(text)
	# WordCloud画像をバイナリ形式で保存
	img = BytesIO()
	wordcloud.to_image().save(img, format='PNG')
	img.seek(0)
	# 画像をBase64エンコード
	img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
	#print(img_base64)
	img_list.append(img_base64)

df['feature_color'] = feature_color
df['img_base64'] = img_list
print(df)
df.to_csv("./nursery_school/nursery_data.csv", index=False)
