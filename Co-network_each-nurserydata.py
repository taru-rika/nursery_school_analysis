import pandas as pd
import numpy as np
import MeCab
import itertools
import collections
from IPython.display import HTML
from pyvis.network import Network
import networkx as nx
from networkx.drawing.nx_pydot import write_dot, graphviz_layout
import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import openai
import json
from sklearn.feature_extraction.text import TfidfVectorizer

# CSVファイルを読み込む
def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df

def update_text_file(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    updated_lines = []
    skip_next = False

    for i in range(len(lines) - 1):
        if skip_next:
            skip_next = False
            continue

        current_line = lines[i].strip()
        next_line = lines[i + 1].strip()

        if '名詞-数詞' in current_line and '名詞-普通名詞-助数詞可能' in next_line:
            current_line_parts = current_line.split()
            next_line_parts = next_line.split()

            if current_line_parts and next_line_parts:
                combined_word = current_line_parts[0] + next_line_parts[0]
                current_line = current_line.replace(current_line_parts[0], combined_word, 1)
                skip_next = True

        updated_lines.append(current_line)

    # 最後の行を確認
    if not skip_next:
        updated_lines.append(lines[-1].strip())

    with open(output_file_path, 'w', encoding='utf-8') as file:
        for line in updated_lines:
            file.write(line + '\n')

def extract_nouns(file_path, output_file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lists = []
        current_list = []
        for line in file:
            if 'EOS' in line:
                if current_list:
                    lists.append(current_list)
                    current_list = []
                continue

            if '名詞' in line:
                first_word = line.split()[0]
                current_list.append(first_word)

        if current_list:
            lists.append(current_list)

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for list_ in lists:
            output_file.write(' '.join(list_) + '\n')

# idfが特定の保育所データのみ使用する形になっているため間違い！！！！！
def tfidf(word_list, output_file_path):
    docs = np.array(word_list)#Numpyの配列に変換する
    ######print(len(docs))

    #単語を配列ベクトル化して、TF-IDFを計算する
    vecs0 = TfidfVectorizer(
                token_pattern=u'(?u)\\b\\w+\\b'#文字列長が 1 の単語を処理対象に含めることを意味します。
                )
    vecs = vecs0.fit_transform(docs)
    terms = vecs0.get_feature_names_out()
    vecs = vecs.toarray()
    ######print(len(vecs))

    vecs = vecs[0]
    top_n_idx = vecs.argsort()[-100:][::-1]
    node_list = [terms[idx] for idx in top_n_idx]
    ######print(node_list)

    words = [i for i in docs if i in node_list]
    # 最初の文書の単語の出現順を維持しつつ復元
    #words_with_freq = vecs0.inverse_transform(vecs)[0]  # 出現した単語のリスト
    #words = " ".join(words_with_freq)  # 元の出現順を維持した文字列
    ######print(words)

# idfをすべての保育所の単語リストとした時
def compute_tfidf(all_word_lists, target_word_list, output_file_path):
    """
    - all_word_lists: すべての保育所の単語リスト（リストのリスト）
    - target_word_list: 特定の保育所の単語リスト
    - output_file_path: 結果の保存先
    """
    # すべての保育所のデータを1つの文書として扱う
    #all_docs = [' '.join(words) for words in all_word_lists]
    ###print("all_docs?????", all_docs)
    ##print("all_word_lists??????", all_word_lists)
    
    # TF-IDF 計算
    vecs0 = TfidfVectorizer(token_pattern=u'(?u)\\b\\w+\\b')
    vecs = vecs0.fit_transform(all_word_lists)
    #vecs = vecs0.fit_transform(all_docs)
    
    # 全体の単語リスト
    terms = vecs0.get_feature_names_out()

    # 特定の保育所データに対するTF-IDFベクトルを計算
    target_doc = ' '.join(target_word_list)
    target_vec = vecs0.transform([target_doc]).toarray()[0]

    # TF-IDF スコアの高い単語を取得
    top_n_idx = target_vec.argsort()[-100:][::-1]  # 上位100単語
    top_words = [terms[idx] for idx in top_n_idx]

    # 結果を保存
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(' '.join(top_words))

    #return words
    #with open(output_file_path, 'w', encoding='utf-8') as output_file:
    #    for word in words:
    #        output_file.write(''.join(word) + ' ')
    #        #output_file.write(' '.join(list_) + '\n')



def create_noun_combinations(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = [line.strip().split() for line in file if line.strip()]

    sentences_combs = [list(itertools.combinations(sentence, 2)) for sentence in sentences]
    return sentences_combs

############ この関数がうまく作動していない！！！！！！
#def count_noun_combinations(input_file_path):
def count_noun_combinations(sentence):
    #with open(input_file_path, 'r', encoding='utf-8') as file:
    #    sentences = [line.split(" ") for line in file]
    #    #sentences = [line.strip().split() for line in file if line.strip()]
    
    # "" を削除
    #sentences = [[word for word in sentence if word != '""'] for sentence in sentences]
    print(">>> sentences の中身例:", sentence[:5])

    sentences = []
    sentences.append(sentence)

    #sentences = sentences[0]
    ######print("This is sentences!!!!!", sentences)

    sentences_combs = [list(itertools.combinations(sentence, 2)) for sentence in sentences]
    sorted_combs = [[tuple(sorted(comb)) for comb in sentence] for sentence in sentences_combs]


    target_combs = []
    for words_comb in sorted_combs:
        target_combs.extend(words_comb)
    # すべての組み合わせをリスト化し、"" を含むものを削除
    #target_combs = [comb for words_comb in sorted_combs for comb in words_comb if '""' not in comb]

    # 単語の組み合わせをカウント
    ct = collections.Counter(target_combs)
    return ct
    ######print(ct.most_common()[:1000])
    #return ct.most_common()[:1000]
    # 組み合わせ回数に閾値を設けて10回以上50回未満のノードのみ表示する
    ######print("This is ct!!!!!", [i for i in ct.items() if (i[1] >= 10 and i[1] <= 50)])
    #return [i for i in ct.items() if (i[1] >= 10 and i[1] <= 50)]

def create_and_save_aggregated_noun_combinations(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        sentences = [line.strip().split() for line in file if line.strip()]

    sentences_combs = [list(itertools.combinations(sentence, 2)) for sentence in sentences]
    sorted_combs = [[tuple(sorted(comb)) for comb in sentence] for sentence in sentences_combs]

    # すべての単語の組み合わせを一つのリストに集約
    target_combs = []
    for words_comb in sorted_combs:
        target_combs.extend(words_comb)

    # 集約されたリストをファイルに保存
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for comb in target_combs:
            output_file.write(f'{comb[0]}, {comb[1]}\n')


import os
import re

def escape_special_chars(text):
    text = re.sub(r'[\\"]', '', text)  # `\` と `"` を削除
    text = text.strip()  # 前後の空白を削除
    return text

# GraphvizのPATHを設定
os.environ["PATH"] += os.pathsep + "/usr/local/bin"  # 必要に応じて変更

def create_cooccurrence_network(most_common_combinations, color_renso):
    # 日本語フォントの設定
    font_path = '/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc'  # 例: macOS の場合
    prop = fm.FontProperties(fname=font_path)

    # ネットワークの初期設定
    cooc_net = nx.Graph()
    #cooc_net = Network(height="750px", width="100%", bgcolor="#FFFFFF", font_color="black")

    # データフレームの作成
    data = pd.DataFrame(most_common_combinations.items(), columns=['word_pair', 'count'])
    # word_pair の最初の数件を確認して、リストやタプルが2要素であるかを確認
    #####print(data['word_pair'].head())
    # 各要素がリストまたはタプルで、かつ2要素であるかをチェック
    #####print(data['word_pair'].apply(lambda x: len(x) == 2 if isinstance(x, (list, tuple)) else False).sum())
    #####print(len(data))  # dataの行数
    #####print(data['word_pair'].isnull().sum())  # 欠損値の数を確認
    #####print(len(data['word_pair'].tolist()))  # word_pairをリストに変換した際の長さ

    data[['word1', 'word2']] = pd.DataFrame(data['word_pair'].tolist(), index=data.index)
    ##########print("This is the length of data!!!!!", len(data))


    ###########print("=== data['word_pair'] の型 ===")
    ###########print(data['word_pair'].apply(type).value_counts())  # データ型の確認

    ###########print("\n=== data['word_pair'] の最初の5件 ===")
    ###########print(data['word_pair'].head())

    ###########print("\n=== data['word_pair'].tolist() の長さ ===")
    ###########print(len(data['word_pair'].tolist()))

    ###########print("\n=== データフレーム化した場合の形状 ===")
    #df_check = pd.DataFrame(data['word_pair'].tolist())
    ###########print(df_check.shape)  # (行数, 列数)
    ###########print(df_check.head())  # 最初の5行を確認
    
    # エッジデータの追加
    for index, row in data.iterrows():
        word1 = row['word1']
        #word1 = escape_special_chars(row['word1'])[:50]
        word2 = row['word2']
        #word2 = escape_special_chars(row['word2'])[:50]
        if word1 and word2:
            cooc_net.add_node(word1, label=word1)
            cooc_net.add_node(word2, label=word2)
            if not cooc_net.has_edge(word1, word2):
                cooc_net.add_edge(word1, word2, value=row['count'])
        #cooc_net.add_node(row['word1'], label=row['word1'], title=row['word1'])
        #cooc_net.add_node(row['word2'], label=row['word2'], title=row['word2'])
        #cooc_net.add_edge(row['word1'], row['word2'], value=row['count'])
    # DOTファイルをデバッグ用に保存
    write_dot(cooc_net, "debug_graph.dot")
    ############print("DOTファイルを 'debug_graph.dot' に保存しました。")

    # ノードの色を設定
    node_colors = {}
    ############print(color_renso)
    for node in cooc_net.nodes:
        ############print(node)
        if node in color_renso.keys():
            sentiment = color_renso[node]
            if sentiment == "ポジティブ":
                node_colors[node] = "blue"
            elif sentiment == "ネガティブ":
                node_colors[node] = "red"
            else:
                node_colors[node] = "gray"
    
    #TODO!!!!!!
    # TDIDF次数に基づいてサイズを設定
    ##node_sizes = {node: cooc_net.degree(node) * 20 for node in cooc_net.nodes} # 倍率を調整可能

    # 保存したノード位置ファイルを読み込む
    with open('mds_node_positions_v3.json', 'r') as f:
        node_positions = json.load(f)
    # 位置情報が存在するノードのみをフィルタリングして描画
    filtered_positions = {node: pos for node, pos in node_positions.items() if node in cooc_net.nodes}
    filtered_nodes = list(filtered_positions.keys())  # 位置情報があるノードのリスト
    ############print(filtered_nodes)
    ############print(filtered_positions)
   
    for node in cooc_net.nodes:
        clean_node = escape_special_chars(node)
        cooc_net.nodes[node]["label"] = clean_node.encode("utf-8").decode("utf-8")

    # Graphvizを使用してノード位置を計算
    #pos = graphviz_layout(cooc_net, prog="dot")  # 'dot', 'neato', 'fdp' などを選択可能
    #try:
    #    pos = nx.spring_layout(cooc_net, k=1.2, seed=42)
    #    #pos = graphviz_layout(cooc_net, prog="dot")
    #except Exception as e:
    #    ############print("Graphviz Error:", e)
    #    raise
    #node_positions = {node: [x, -y] for node, (x, y) in pos.items()}  # Y軸反転

    # ネットワークの初期設定
    cooc_net_pyvis = Network(height="750px", width="100%", bgcolor="#FFFFFF", font_color="black")
    #cooc_net_pyvis.from_nx(cooc_net)
   
    # ノード位置を設定
    for node in filtered_positions:
    #for node in cooc_net.nodes:
        #if node in node_colors.keys():   ##TODO!!!!! delete this sentence!!!!!!
            x, y = filtered_positions[node]

            #TODO!!! node_sizes
            #cooc_net_pyvis.add_node(node, x=x, y=y, fixed=True, size=node_sizes[node], color=node_colors[node])
            cooc_net_pyvis.add_node(node, x=x, y=y, fixed=True, size=5, color=node_colors[node])
    # エッジデータの追加
    for index, row in data.iterrows():
        if row['word1'] in cooc_net_pyvis.get_nodes() and row['word2'] in cooc_net_pyvis.get_nodes():
            cooc_net_pyvis.add_edge(row['word1'], row['word2'], value=row['count'], color={'highlight': 'red', 'hover': 'red'})

    # 最大エッジ重み取得（スライダー最大値用）
    max_weight = max(most_common_combinations.values()) if most_common_combinations else 1

    # 共起ネットワークを描画（位置情報があるノードのみ）
    #nx.draw(cooc_net.subgraph(filtered_nodes), pos=filtered_positions, font_family=prop.get_name(), with_labels=True)
    #plt.show()

    # ネットワークの可視化設定
    cooc_net_pyvis.set_options("""
    var options = {
      "interaction": {
        "hover": true
      },
      "nodes": {
        "font": {
          "size": 15
        },
        "scaling": {
          "min" : 10,
          "max" : 30 
        },
        "color": {
          "highlight": {
            "border": "red",
            "background": "red"
          }
        }
      },
      "edges": {
        "color": {
          "highlight": "red",
          "hover": "red"
        },
        "smooth": false 
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -26,
          "centralGravity": 0.005,
          "springLength": 230,
          "springConstant": 0.18
        },
        "maxVelocity": 1,
        "solver": "forceAtlas2Based",
        "timestep": 0.01,
        "stabilization": {
          "enabled": true,
          "iterations": 2000,
          "updateInterval": 25
        }
      }
    }
    """)
    
    return cooc_net_pyvis, max_weight


# 形態素解析モデルの構築
t = MeCab.Tagger()

# CSVファイルのパスと解析したい列の名前を指定
file_path = './nursery_school/nursery_data2.csv'  # CSVファイルのパス
#column_name = 'analyze_service'   # 解析したい列の名前

# CSVファイルから文章を読み込む
df = read_csv(file_path)
#sentences = read_csv(file_path, column_name)

# 施設名ごとにグループ化
groups = df.groupby('facilityName')

# 出力ファイルの設定
output_dir = './nursery_school/MeCab_EachNurserySchool/mecab_example'
output_dir2 = './nursery_school/MeCab_EachNurserySchool/sentences'
output_dir3 = './nursery_school/new_negapoji/negapoji'
output_dir4 = './nursery_school/youyaku/negapoji'

# 形態素解析の実行とファイルへの書き込み
specific_string = "共" + "通" + "評" + "価" + "項" + "目" + "\\" + "n" + "\\" + "t" + "\\" + "t" + "\\" + "t" + "\\" + "t"
openai.api_key = "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" 
all_word_txt = output_dir + ".txt"
with open(all_word_txt, 'w', encoding='utf-8') as file:
    tokens_list = []
    for name, group in groups:
        if(name != "山内保育所" and name != "越来保育所" and name != "知花保育所"):
            sentences = group['analyze_service'].tolist()
            for sentence in sentences:
                sentence = str(sentence).lower()
                sentence = sentence.replace('[','')
                sentence = sentence.replace(']','')
                sentence = sentence.replace(',','')
                start_index = sentence.find(specific_string)
                sentence = sentence[start_index + len(specific_string):]
                tokens = t.parse(sentence)
                tokens_list.append(tokens)
                file.write(tokens)
# 形態素解析結果をアップデートする
all_output_file_path = output_dir + "_re_" + name  + '.txt'  # 出力ファイルのパス
update_text_file(all_word_txt, all_output_file_path)

# 名詞のみ抽出する
#all_output_file_path2 = './nursery_school/Noun_EachNurserySchool/meishi_example_' + name + '.txt'  # 出力ファイルのパスを設定してください
#extract_nouns(all_output_file_path, all_output_file_path2)
f_all = open('./meishi_example.txt', 'r')
all_meishis_example = f_all.read()
all_meishis_example = all_meishis_example.split()
all_meishis_example = [dt for dt in all_meishis_example if dt != '""' and dt != '"']
###print("all_meishis_example?????", all_meishis_example)

for name, group in groups:
    if(name != "山内保育所" and name != "越来保育所" and name != "知花保育所"):
        sentences = group['analyze_service'].tolist()
        output_file = output_dir + "_" + name + ".txt"
        # 文章データを保存する
        output_file2 = output_dir2 + "_" + name + ".txt"
        with open(output_file, 'w', encoding='utf-8') as file:
            with open(output_file2, 'w', encoding='utf-8') as file1:
                for sentence in sentences:
                    sentence = str(sentence).lower()
                    sentence = sentence.replace('[','')
                    sentence = sentence.replace(']','')
                    sentence = sentence.replace(',','')
                    start_index = sentence.find(specific_string)
                    sentence = sentence[start_index + len(specific_string):]
                    tokens = t.parse(sentence)
                    file.write(tokens)
                    file1.write(sentence)

        #########print(f"形態素解析結果を {output_file} に保存しました。")
        #########print(f"sentenceデータを {output_file2} に保存しました。")
        ####print("tokens?????", tokens)

        # 形態素解析結果をアップデートする
        output_file_path = output_dir + "_re_" + name  + '.txt'  # 出力ファイルのパス
        update_text_file(output_file, output_file_path)

        # 名詞のみ抽出する
        output_file_path2 = './nursery_school/Noun_EachNurserySchool/meishi_example_' + name + '.txt'  # 出力ファイルのパスを設定してください
        extract_nouns(output_file_path, output_file_path2)
        f1 = open(output_file_path2, 'r')
        meishis_example = f1.read()
        meishis_example = meishis_example.split()
        meishis_example = [dt for dt in meishis_example if dt != '""' and dt != '"']
        #print("This is meishis_example!!!!", meishis_example[:10])
        # TF-IDFで上位30件の名詞を表示する
        output_file_path6 = './nursery_school/Noun_EachNurserySchool/meishi_tfidf_' + name + '.txt' 
        #tfidf(meishis_example, output_file_path6)
        compute_tfidf(all_meishis_example, meishis_example, output_file_path6)
        f2 = open(output_file_path6, 'r')
        tfidf_words = set(f2.read().split())
        meishis = [dt for dt in meishis_example if dt in tfidf_words]
        #print("This is meishis!!!!", meishis[:10])
        f3 = open(output_file2, 'r')
        sentences = f3.read()
        sentences = sentences.split('""')
        sentences = sentences[1:] 
        sentences = [dt for dt in sentences if dt != '""' and dt != '"']
        meishi_bunsho = {}
        for meishi in meishis:
            bunsho = []
            for sentence in sentences:
              if( (meishi in sentence) == True):
                  bunsho.append(sentence)
            meishi_bunsho[meishi] = bunsho
        #######print("This is meishi_bunsho!!!!!", meishi_bunsho)

        # 抽出した名詞がプラスマイナスどちらの意味を持つか調べる
        ### 原文のままネガポジ分析した場合
        output_file3 = output_dir3 + "_" + name + "_tfidf_v3.json"
        # 課金エリア！！！！！！気をつけろ！！！！！
        if not os.path.isfile(output_file3):
            with open(output_file3, 'w', encoding='utf-8') as file:
                negapoji = {}
                for key in meishi_bunsho.keys():
                    ############print(key)
                    #########print(list(meishi_bunsho[key]))
                    ###  ここにChatGPTを使用したネガポジ分析のコードを付け加えたい！！！！
                    res = openai.ChatCompletion.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": "日本語で返答してください。"   
                            },
                            {
                                "role": "system",
                                "content": "単語のみで返答してください。\n丁寧語はやめてください。\n 句読点もやめてください。"
                            },
                            {
                                "role": "system",
                                "content": f"{list(meishi_bunsho[key])}の要素1つずつ分析します。\n ポジティブであれば1点、ネガティブであれば-1点です。\n 配列の合計値がプラスであればポジティブ、マイナスであればネガティブです。"
                            },
                            {
                                "role": "user",
                                #"content": f"{meishi_bunsho[key]}に含まれる{key}はポジティブですか。ネガティブですか。"
                                #"content": f"{meishi_bunsho[key]}はポジティブですか。ネガティブですか。"
                                "content": "ポジティブですか。ネガティブですか。"
                            }
                        ]
                    )
                    print(res["choices"][0]["message"]["content"])
                    negapoji[key] = res["choices"][0]["message"]["content"]
                json.dump(negapoji, file, ensure_ascii=False, indent=4)  # 日本語をそのまま保存

        f = open(output_file3, 'r')
        #f = open(output_file4, 'r')
        negapoji = f.read()
        negapoji = json.loads(negapoji)
 
        # 名詞の組み合わせを作成する
        #noun_combinations = create_noun_combinations(output_file_path6)
        #############print(noun_combinations[0])  # 最初の文の名詞の組み合わせを表示

        # 名詞の組み合わせ数を数える
        #most_common_combinations = count_noun_combinations(output_file_path6)
        #print(most_common_combinations)

        # 名詞の組み合わせと組み合わせ数をファイルに保存する
        #output_file_path3 = './nursery_school/Kyoki_gyoretsu_EachNurserySchool/kyoki_gyoretsu_' + name + '.txt'  # 出力ファイルのパスを設定してください
        #create_and_save_aggregated_noun_combinations(output_file_path6, output_file_path3)

        # この関数を使用するには、most_common_combinationsを引数として渡します
        output_file_path5 = './Kyoki-Network_HTML/cooccurrence_network_' + name + '_tfidf_v2.html'
        if(os.path.isfile(output_file_path6)):
            most_common_combinations = count_noun_combinations(meishis)
            print("=== most_common_combinations の最初の5件 ===")
            print(most_common_combinations)
            #####print(negapoji)
            print(len(most_common_combinations))
            #####print(len(negapoji))

            cooccurrence_network, max_weight = create_cooccurrence_network(most_common_combinations, negapoji)
            # HTMLファイルに保存してインタラクティブ表示を有効化
            cooccurrence_network.write_html(output_file_path5, local=True)
    
            with open(output_file_path5, "r", encoding="utf-8") as file:
                html_content = file.read()

            sentence_js = f"var sentenceData = {json.dumps(sentences, ensure_ascii=False)};"
            
            # 移動用: ネットワーク描画部分を抽出して後で移動
            start_div = html_content.find('<div id="mynetwork"')
            end_div = html_content.find('</script>', start_div) + len('</script>')
            mynetwork_block = html_content[start_div:end_div]

            # <body> 内にノード情報表示用の <div> を追加
            new_body = f'''
            <div style="margin-bottom: 10px;">
              <label for="weightSlider">共起回数（エッジの重み）以上を表示: </label>
              <input type="range" id="weightSlider" min="1" max="{max_weight}" value="1">
              <span id="weightValue">1</span>
            </div>
            <div id="nodeInfo" style="margin-bottom: 20px; font-size: 16px; font-weight: bold;">選択したノードの情報がここに表示されます</div>
            <div id="container" style="display: flex; flex-direction: row;">
              <div id="relatedSentences" style="width: 30%; padding: 10px; border: 1px solid #ccc; background: #f9f9f9; font-size: 14px; overflow-y: auto; height: 750px;">
                <strong>関連文:</strong><br>
              </div>
              <div style="width: 70%; padding-left: 10px;">
                {mynetwork_block}
              </div>
            </div>
            <script>{sentence_js}</script>
            '''
            # <body>〜</body> の中身を書き換える
            start_body = html_content.find('<body>') + len('<body>')
            end_body = html_content.find('</body>')
            new_html = html_content[:start_body] + new_body + html_content[end_body:]

            # JavaScriptでクリック時に情報を更新
            js_script = f"""
            <script type="text/javascript">
            document.addEventListener("DOMContentLoaded", function() {{
                var originalNodes = nodes.get();
                var originalEdges = edges.get();

                // スライダーUI
                var slider = document.getElementById("weightSlider");
                var weightValue = document.getElementById("weightValue");

                slider.addEventListener("input", function() {{
                    var minWeight = parseInt(slider.value);
                    weightValue.innerText = minWeight;
                    updateNetwork(minWeight);
                }});

                function updateNetwork(minWeight) {{
                    var filteredEdges = originalEdges.filter(e => e.value >= minWeight);
                    var usedNodeIds = new Set();
                    filteredEdges.forEach(e => {{
                        usedNodeIds.add(e.from);
                        usedNodeIds.add(e.to);
                    }});
                    var filteredNodes = originalNodes.filter(n => usedNodeIds.has(n.id));
                    var data = {{ nodes: new vis.DataSet(filteredNodes), edges: new vis.DataSet(filteredEdges) }};
                    network.setData(data);
                }}

                // ノードクリック時の情報表示
                network.on("click", function(params) {{
                    if (params.nodes.length > 0) {{
                        var selectedNode = params.nodes[0];
                        var connectedNodes = network.getConnectedNodes(selectedNode);
                        var text = "選択ノード: " + selectedNode + "<br>接続ノード: " + connectedNodes.join(", ");
                        document.getElementById("nodeInfo").innerHTML = text;
                
                        var related = sentenceData.filter(s => s.includes(selectedNode));
                        var highlighted = related.map(s => s.replaceAll(selectedNode, `<span style=\"color:red\">${{selectedNode}}</span>`));
                        document.getElementById("relatedSentences").innerHTML = "<strong>関連文:</strong><br>" + highlighted.join("<br><br>");
                    }}
                }});
            }});
            </script>            
            """

            new_html = new_html.replace("</body>", js_script + "\n</body>")

            # 修正した HTML を上書き保存
            with open(output_file_path5, "w", encoding="utf-8") as file:
                file.write(new_html)

        f1.close() 
        f2.close() 




