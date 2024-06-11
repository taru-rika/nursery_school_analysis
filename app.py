import requests
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster
import pandas as pd
import numpy as np
import urllib

# ページ設定
st.set_page_config(
    page_title="幼稚園・保育園選びお助けアプリ",
    page_icon="🚼",
    layout="wide"
)

pages = ["search", "recommend"]
page = st.sidebar.radio("""### menu""",pages)
facility_list =[]

if page == "search":
	st.title("専門家向け保育園・幼稚園評価分析システム")
	st.markdown("""#### 概要：
		教育の専門家や自治体が保育園・幼稚園を分析する際のサポートシステムとなることを目標とします""")
	
#	st.markdown("---")
#	st.markdown("""#### 検索条件設定""")
#	col1, col2, col3 = st.columns(3)
#
#	with col1:
#		st.markdown("""#### 1 地点""")
#		plc=st.text_input("以下に入力してください.ex)茗荷谷駅",value="")
#	
#	#半径を指定する場合
#	with col2:
#		st.markdown("""#### 2 範囲""")
#		plc_range = st.slider('地点を中心とした距離(m)',value=500, min_value=0, max_value=3000) #スライダーをつける
#		st.write("地点からの直線距離{:,}m".format(plc_range))
#		
#		#施設の種類
#		#facility_type=[
#		#	"認可保育所",
#		#	"認証保育所A型・B型",
#		#	"認定こども園",
#		#	"認可外保育施設",
#		#	"乳児院",
#		#	"児童養護施設",
#		#	"児童自立生活援助事業[自立援助ホーム]",
#		#	"母子生活支援施設",
#		#	"児童自立支援施設",
#		#]
#		facility = pd.read_csv('./nursery_school/nursery_data.csv')
#		facility_type = np.array(facility['事業所種'])
#		facility_type = set(facility_type)
#		print(facility_type)
#		facility_list.append(facility['事業所名称'])
#
#	with col3:
#		st.markdown("""#### 3 施設の種類""")
#		ty=st.selectbox("以下から一つ選んでください", facility_type)
#		st.write('選択したオプション：',ty)
#
#	button = st.button('検索')
#	st.markdown("---")

	m = folium.Map(
	    # 地図の中心位置の指定(今回は東京駅を指定)
	    location=[35.72 , 139.75], 
	    # タイル、アトリビュートの指定
	    tiles='https://cyberjapandata.gsi.go.jp/xyz/airphoto/{z}/{x}/{y}.png',
	    attr= '(令和5年度10月)',
	    #attr= ty + '(令和5年度10月)',
	    # ズームを指定
	    zoom_start=15
	)
	
	#マップに検索結果を表示する
	# 地図の中心の緯度/経度、タイル、初期のズームサイズを指定します。
	
	print("facility_list : ", facility_list)
	#表示するデータを読み込み
	df = pd.read_csv('./nursery_school/nursery_data.csv')
	#df = pd.read_csv('./nursery_school/202310-2-1_' + ty +'.csv', encoding="shift-jis", nrows=10)
	
	# 読み込んだデータ(緯度・経度、ポップアップ用文字、アイコンを表示)
	marker_cluster = MarkerCluster()
	for i, row in df.iterrows():
	    # ポップアップの作成
	    #pop = f'<img src= "data:image/png;base64,{row['img_base64']}" >'
	    pop = f'{row["事業所名称"]} <img src="data:image/png;base64,{row["img_base64"]}">'
	    folium.Marker(
	        # 緯度と経度を指定
                location=[row['lat'],row['lon']],
	        # ツールチップの指定(都道府県名)
	        tooltip=row['事業所名称'],
	        # ポップアップの指定
	        popup=folium.Popup(pop, max_width=400),
	        # アイコンの指定(アイコン、色)
	        #icon=folium.Icon(color="red")
	        icon=folium.Icon(icon='home', icon_color=row['feature_color'], color='white')
	    ).add_to(marker_cluster)
	marker_cluster.add_to(m)
	st_data = st_folium(m, width=1200, height=800)
else:
	st.title("幼稚園・保育園選びお助けアプリ")
	st.markdown("""#### 概要：
		あなたと幼稚園（保育園）両者のニーズにマッチした幼稚園（保育園）をレコメンドします．""")
	

