import requests
import os.path
import urllib
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np

##保育園の口コミ・カラーデータの読み込み
nursery_data = pd.read_csv('./unlicensed_nursery_data.csv')
#nursery_list = np.array(nursery_data['事業所名称'])
##保育園の座標データの読み込み
#ty = '認可保育所'
#df = pd.read_csv('./202310-2-1_' + ty +'.csv', encoding="shift-jis")
#
#nursery_data['address']=''
#nursery_data['lon']=''
#nursery_data['lat']=''
#page_num = 40
#
##for i, row in nursery_data.iterrows():
##    for j, row2 in df.iterrows():
##    	if row['事業所名称']==row2['施設名'] :
##            #row['address'] = row2['所在地']
##            #print(row['address'])
##            nursery_data.at[i, 'address'] += row2['所在地']
##    print(nursery_data)
##
##nursery_data.to_csv("./nursery_data.csv", index=False)
#nursery_data = pd.read_csv('./nursery_data.csv')

#読み込んだデータ(緯度・経度、ポップアップ用文字、アイコンを表示)
for i, row in nursery_data.iterrows():
    #print(row)
    if row['施設所在地'] != np.nan:
        address = row['施設所在地']
        print(address)
        makeUrl = "https://msearch.gsi.go.jp/address-search/AddressSearch?q="
        s_quote = urllib.parse.quote(address)
        response = requests.get(makeUrl + s_quote)
        location=response.json()[0]["geometry"]["coordinates"]
        lonlat = [location[0],location[1]]
        print(lonlat)
        nursery_data.at[i, 'lon'] = location[0]
        nursery_data.at[i, 'lat'] = location[1]
        #print(nursery_data)
        nursery_data.to_csv("./unlicensed_nursery_data.csv", index=False)
