from bertopic import BERTopic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

model = BERTopic(embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 文書のリストを用意しておく
df = pd.read_csv('./nursery_school/nursery_data2.csv' ,skip_blank_lines=True)
#df = pd.read_csv('./nursery_school/unlicensed_nursery_data.csv' ,skip_blank_lines=True)
df.index = range(len(df))
df = df.dropna(how='all')  # すべての列がNaNである行を削除
df = df.dropna(subset=['analyze_service'])
#print(df)
#text_list = np.array(df['方針・理念'].astype(str) + ' ' + df['teacher'].astype(str) + ' ' + df['保育・教育内容'].astype(str))
text_list = np.array(df['analyze_service'])
print(text_list)
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
	
print(text_new_list)
print(len(text_new_list))

#news_data: list[str] = preprocess_data()
topics, probs = model.fit_transform(text_new_list)

# トピックの可視化を行う
# 1. トピックのバーチャートを表示
fig1 = model.visualize_barchart()
fig1.show()
# 2. それぞれのトピックの単語の重要度をみる場合
fig2 = model.get_topic(0)
print(fig2)
# 3. トピックのヒートマップを表示
fig3 = model.visualize_heatmap()
fig3.show()
