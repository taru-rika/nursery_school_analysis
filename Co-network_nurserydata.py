import pandas as pd
import MeCab
import time
from sklearn.manifold import MDS
import numpy as np
import json
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt

# CSVファイルを読み込む
def read_csv(file_path, column_name):
    df = pd.read_csv(file_path)
    return df[column_name].tolist()

# 形態素解析モデルの構築
t = MeCab.Tagger()

# CSVファイルのパスと解析したい列の名前を指定
file_path = './nursery_school/nursery_data2.csv'  # CSVファイルのパス
column_name = 'analyze_service'   # 解析したい列の名前

# CSVファイルから文章を読み込む
sentences = read_csv(file_path, column_name)

# 出力ファイルの設定
output_file = './mecab_example.txt'

# 形態素解析の実行とファイルへの書き込み
specific_string = "共" + "通" + "評" + "価" + "項" + "目" + "\\" + "n" + "\\" + "t" + "\\" + "t" + "\\" + "t" + "\\" + "t"
with open(output_file, 'w', encoding='utf-8') as file:
    for sentence in sentences:
        sentence = str(sentence).lower()
        sentence = sentence.replace('[','')
        sentence = sentence.replace(']','')
        sentence = sentence.replace(',','')
        start_index = sentence.find(specific_string)
        sentence = sentence[start_index + len(specific_string):]
        print(sentence)
        tokens = t.parse(sentence)
        file.write(tokens)

print(f"形態素解析結果を {output_file} に保存しました。")

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

# 使用例
input_file_path = 'mecab_example.txt'  # 入力ファイルのパス
output_file_path = 'mecab_example_re.txt'  # 出力ファイルのパス
update_text_file(input_file_path, output_file_path)

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

# この関数を使ってファイルを解析し、結果を別のファイルに保存するには、以下のようにします
input_file_path = 'mecab_example_re.txt'  # 入力ファイルのパスを設定してください
output_file_path = 'meishi_example.txt'  # 出力ファイルのパスを設定してください
extract_nouns(input_file_path, output_file_path)

import itertools
def create_noun_combinations(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = [line.strip().split() for line in file if line.strip()]

    sentences_combs = [list(itertools.combinations(sentence, 2)) for sentence in sentences]
    return sentences_combs

# この関数を使用する例
file_path = 'meishi_example.txt'  # 名詞が格納されたファイルのパスを設定してください
noun_combinations = create_noun_combinations(file_path)
print(noun_combinations[0])  # 最初の文の名詞の組み合わせを表示

import itertools
import collections

def count_noun_combinations(input_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        sentences = [line.strip().split() for line in file if line.strip()]

    sentences_combs = [list(itertools.combinations(sentence, 2)) for sentence in sentences]
    sorted_combs = [[tuple(sorted(comb)) for comb in sentence] for sentence in sentences_combs]

    target_combs = []
    for words_comb in sorted_combs:
        target_combs.extend(words_comb)

    # 単語の組み合わせをカウント
    ct = collections.Counter(target_combs)
    return ct.most_common()[:1000]

# この関数を使用する例
input_file_path = 'meishi_example.txt'  # 入力ファイルのパスを設定してください
most_common_combinations = count_noun_combinations(input_file_path)
print(most_common_combinations)


import itertools

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

# この関数を使用する例
input_file_path = 'meishi_example.txt'  # 入力ファイルのパスを設定してください
output_file_path = 'kyoki_gyoretsu_example.txt'  # 出力ファイルのパスを設定してください
create_and_save_aggregated_noun_combinations(input_file_path, output_file_path)

from IPython.display import HTML
HTML("/content/cooccurrence_network.html")
from pyvis.network import Network
import pandas as pd

def create_cooccurrence_network(most_common_combinations):
    # ネットワークの初期設定
    cooc_net = nx.Graph()
    #cooc_net = Network(height="750px", width="100%", bgcolor="#FFFFFF", font_color="black")

    # データフレームの作成
    data = pd.DataFrame(most_common_combinations, columns=['word_pair', 'count'])
    data[['word1', 'word2']] = pd.DataFrame(data['word_pair'].tolist(), index=data.index)

    # エッジデータの追加
    for index, row in data.iterrows():
        #cooc_net.add_node(row['word1'], label=row['word1'], title=row['word1'])
        #cooc_net.add_node(row['word2'], label=row['word2'], title=row['word2'])
        cooc_net.add_edge(row['word1'], row['word2'], weight=row['count'])
    
    # 単語のリストを作成
    #words = list(set(data['word1'].tolist() + data['word2'].tolist()))

    # 共起行列の作成
    #word_index = {word: idx for idx, word in enumerate(words)}
    #cooccurrence_matrix = np.zeros((len(words), len(words)))

    #for _, row in data.iterrows():
    #    i = word_index[row['word1']]
    #    j = word_index[row['word2']]
    #    cooccurrence_matrix[i, j] = row['count']
    #    cooccurrence_matrix[j, i] = row['count']

    # ノードの次数に基づいてサイズを設定
    node_sizes = {node: cooc_net.degree(node) * 50 for node in cooc_net.nodes} # 倍率を調整可能
    #degrees = dict(cooc_net.degree())  # ノードの次数（接続エッジ数）
    #max_degree = max(degrees.values()) if degrees else 1  # 最大次数を取得
    #node_sizes = {node: (degree / max_degree) * 50 + 10 for node, degree in degrees.items()}  # サイズのスケーリング
    print(node_sizes)

    
    # MDSによるノード位置の固定
    words = list(cooc_net.nodes)
    word_index = {word: idx for idx, word in enumerate(words)}
    cooccurrence_matrix = np.zeros((len(words), len(words)))
    cooccurrence_matrix = nx.to_numpy_array(cooc_net, nodelist=words, weight='weight')
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress=False)
    pos = mds.fit_transform(1 / (cooccurrence_matrix + 1e-5))  # 類似度を距離に変換

    # ノードが重ならないように力学シミュレーションで調整
    positions = {words[i]: pos[i] for i in range(len(words))}
    for _ in range(100):  # 繰り返し計算で位置調整
        for node1, pos1 in positions.items():
            for node2, pos2 in positions.items():
                if node1 != node2:
                    dist = np.linalg.norm(pos1 - pos2)
                    if dist < 0.1:  # 閾値以下の場合は調整
                        direction = (pos1 - pos2) / dist
                        positions[node1] += direction * 0.01
                        positions[node2] -= direction * 0.01
    
    # エッジデータの追加
    for word in words:
        print(node_sizes[word])
        x, y = positions[word]
        #idx = word_index[word]
        cooc_net.add_node(word, label=word, title=word, x=x * 1000, y=y * 1000, size=node_sizes[word])

    for _, row in data.iterrows():
        cooc_net.add_edge(row['word1'], row['word2'], value=row['count'], width=row['count'] * 10)

    # ノード位置を辞書形式で保存
    node_positions = {node: pos[i].tolist() for i, node in enumerate(cooc_net.nodes())}
    # ノード位置をJSONファイルに保存
    with open('mds_node_positions.json', 'w') as f:
        json.dump(node_positions, f, ensure_ascii=False)
    print("ノード位置をmds_node_positions.jsonに保存しました!!!!!")

    # ネットワークの初期設定
    cooc_net_pyvis = Network(height="750px", width="100%", bgcolor="#FFFFFF", font_color="black")
    cooc_net_pyvis.from_nx(cooc_net)

    # ノードとエッジをPyvisネットワークに追加
    for node in cooc_net.nodes:
        x, y = positions[node]
        cooc_net_pyvis.add_node(
            node,
            label=node,
            x=x * 1000,  # スケーリング調整
            y=y * 1000,
            value=node_sizes[node],
        )

    for edge in cooc_net.edges(data=True):
        cooc_net_pyvis.add_edge(edge[0], edge[1], value=edge[2].get('weight', 1))

    # ネットワークの可視化設定
    cooc_net_pyvis.set_options("""
    var options = {
      "nodes": {
        "scaling": {
          "min": 10,
          "max": 5000
        }
      },
      "font": {
          "size": 12
        },
      "edges": {
        "color": {
          "inherit": true
        },
        "smooth": false
      },
      "physics": {
        "enabled": false 
      }
    }
    """)
 
    return cooc_net_pyvis

# この関数を使用するには、most_common_combinationsを引数として渡します
input_file_path = 'meishi_example.txt'  # 入力ファイルのパスを設定してください
most_common_combinations = count_noun_combinations(input_file_path)
cooccurrence_network = create_cooccurrence_network(most_common_combinations)
cooccurrence_network.show("cooccurrence_network.html")

from IPython.display import HTML
HTML("/content/cooccurrence_network.html")


