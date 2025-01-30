import pandas as pd
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

def create_noun_combinations(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = [line.strip().split() for line in file if line.strip()]

    sentences_combs = [list(itertools.combinations(sentence, 2)) for sentence in sentences]
    return sentences_combs

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

def escape_special_chars(text):
    return text.replace('"', '\\"')  # ダブルクオートをエスケープ

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
    data = pd.DataFrame(most_common_combinations, columns=['word_pair', 'count'])
    data[['word1', 'word2']] = pd.DataFrame(data['word_pair'].tolist(), index=data.index)

    # エッジデータの追加
    for index, row in data.iterrows():
        word1 = escape_special_chars(row['word1'])[:50]
        word2 = escape_special_chars(row['word2'])[:50]
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
    print("DOTファイルを 'debug_graph.dot' に保存しました。")

    # ノードの色を設定
    node_colors = {}
    print(color_renso)
    for node in cooc_net.nodes:
        print(node)
        sentiment = color_renso[node]
        if sentiment == "ポジティブ":
            node_colors[node] = "blue"
        elif sentiment == "ネガティブ":
            node_colors[node] = "red"
        else:
            node_colors[node] = "gray"
    
    # ノードの次数に基づいてサイズを設定
    node_sizes = {node: cooc_net.degree(node) * 50 for node in cooc_net.nodes} # 倍率を調整可能

    # 保存したノード位置ファイルを読み込む
    with open('mds_node_positions.json', 'r') as f:
        node_positions = json.load(f)
    # 位置情報が存在するノードのみをフィルタリングして描画
    filtered_positions = {node: pos for node, pos in node_positions.items() if node in cooc_net.nodes}
    filtered_nodes = list(filtered_positions.keys())  # 位置情報があるノードのリスト
    print(filtered_nodes)
    print(filtered_positions)
   
    for node in cooc_net.nodes:
        clean_node = escape_special_chars(node)
        cooc_net.nodes[node]["label"] = clean_node.encode("utf-8").decode("utf-8")

    # Graphvizを使用してノード位置を計算
    #pos = graphviz_layout(cooc_net, prog="dot")  # 'dot', 'neato', 'fdp' などを選択可能
    #try:
    #    pos = nx.spring_layout(cooc_net, k=1.2, seed=42)
    #    #pos = graphviz_layout(cooc_net, prog="dot")
    #except Exception as e:
    #    print("Graphviz Error:", e)
    #    raise
    #node_positions = {node: [x, -y] for node, (x, y) in pos.items()}  # Y軸反転

    # ネットワークの初期設定
    cooc_net_pyvis = Network(height="750px", width="100%", bgcolor="#FFFFFF", font_color="black")
    #cooc_net_pyvis.from_nx(cooc_net)
   
    # ノード位置を設定
    for node in filtered_positions:
    #for node in cooc_net.nodes:
        #if node in cooc_net.nodes:
            print(node)
            #x, y = node_positions[node]
            x, y = filtered_positions[node]
            #cooc_net_pyvis.nodes[node]['x'] = x
            #cooc_net_pyvis.nodes[node]['y'] = y
            cooc_net_pyvis.add_node(node, x=x, y=y, fixed=True, size=node_sizes[node], color=node_colors[node])
            #cooc_net_pyvis.add_node(node, x=x, y=y, fixed=True, size=node_sizes[node], color={'highlight': 'red', 'hover': 'red'})
    # エッジデータの追加
    for index, row in data.iterrows():
        if row['word1'] in cooc_net_pyvis.get_nodes() and row['word2'] in cooc_net_pyvis.get_nodes():
            cooc_net_pyvis.add_edge(row['word1'], row['word2'], value=row['count'], color={'highlight': 'red', 'hover': 'red'})


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
          "size": 1000
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
    
    return cooc_net_pyvis


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
output_dir3 = './nursery_school/negapoji/negapoji'

# 形態素解析の実行とファイルへの書き込み
specific_string = "共" + "通" + "評" + "価" + "項" + "目" + "\\" + "n" + "\\" + "t" + "\\" + "t" + "\\" + "t" + "\\" + "t"
openai.api_key = "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" 
for name, group in groups:
    sentences = group['analyze_service'].tolist()
    output_file = output_dir + "_" + name + ".txt"
    with open(output_file, 'w', encoding='utf-8') as file:
        for sentence in sentences:
            sentence = str(sentence).lower()
            sentence = sentence.replace('[','')
            sentence = sentence.replace(']','')
            sentence = sentence.replace(',','')
            start_index = sentence.find(specific_string)
            sentence = sentence[start_index + len(specific_string):]
            tokens = t.parse(sentence)
            file.write(tokens)

    # 文章データを保存する
    output_file2 = output_dir2 + "_" + name + ".txt"
    with open(output_file2, 'w', encoding='utf-8') as file:
        for sentence in sentences:
            sentence = str(sentence).lower()
            sentence = sentence.replace('[','')
            sentence = sentence.replace(']','')
            sentence = sentence.replace(',','')
            start_index = sentence.find(specific_string)
            sentence = sentence[start_index + len(specific_string):]
            file.write(sentence)

    print(f"形態素解析結果を {output_file} に保存しました。")
    print(f"sentenceデータを {output_file2} に保存しました。")

    # 形態素解析結果をアップデートする
    output_file_path = output_dir + "_re_" + name  + '.txt'  # 出力ファイルのパス
    update_text_file(output_file, output_file_path)

    # 名詞のみ抽出する
    output_file_path2 = './nursery_school/Noun_EachNurserySchool/meishi_example_' + name + '.txt'  # 出力ファイルのパスを設定してください
    extract_nouns(output_file_path, output_file_path2)

    # 抽出した名詞がプラスマイナスどちらの意味を持つか調べる
    f1 = open(output_file_path2, 'r')
    f2 = open(output_file2, 'r')
    meishis = f1.read()
    meishis = meishis.split()
    meishis = [dt for dt in meishis if dt != '""' and dt != '"']
    sentences = f2.read()
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
    print(meishi_bunsho)

    output_file3 = output_dir3 + "_" + name + ".json"
    # 課金エリア！！！！！！気をつけろ！！！！！
    with open(output_file3, 'w', encoding='utf-8') as file:
        negapoji = {}
        for key in meishi_bunsho.keys():
            print(key)
            print(list(meishi_bunsho[key]))
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
    negapoji = f.read()
    negapoji = json.loads(negapoji)
    print(negapoji)
 
    # 名詞の組み合わせを作成する
    noun_combinations = create_noun_combinations(output_file_path2)
    #print(noun_combinations[0])  # 最初の文の名詞の組み合わせを表示

    # 名詞の組み合わせ数を数える
    most_common_combinations = count_noun_combinations(output_file_path2)
    print(most_common_combinations)

    # 名詞の組み合わせと組み合わせ数をファイルに保存する
    output_file_path3 = './nursery_school/Kyoki_gyoretsu_EachNurserySchool/kyoki_gyoretsu_' + name + '.txt'  # 出力ファイルのパスを設定してください
    create_and_save_aggregated_noun_combinations(output_file_path2, output_file_path3)

    output_file_path4 = '/content/cooccurrence_network_' + name + '.html'
    HTML(output_file_path4)

    # この関数を使用するには、most_common_combinationsを引数として渡します
    output_file_path5 = './Kyoki-Network_HTML/cooccurrence_network_' + name + '.html'
    most_common_combinations = count_noun_combinations(output_file_path2)
    cooccurrence_network = create_cooccurrence_network(most_common_combinations, negapoji)
    # HTMLファイルに保存してインタラクティブ表示を有効化
    cooccurrence_network.write_html(output_file_path5)

    # カスタムHTMLを追加
    custom_html = """
    <script type="text/javascript">
        document.addEventListener('DOMContentLoaded', function() {
          // Wait for the network to be fully initialized
          setTimeout(function() {
            // Modify the network object to handle node selection
            network.on("selectNode", function(properties) {
              var selectedNodeId = properties.nodes[0];
              var selectedNode = network.body.data.nodes.get(selectedNodeId);
              var connectedNodes = network.getConnectedNodes(selectedNodeId);
              var connectedNodeLabels = connectedNodes.map(function(nodeId) {
                return network.body.data.nodes.get(nodeId).label;
              });

              var infoDiv = document.getElementById("node-info");
              infoDiv.innerHTML = "<strong>選択したノード:</strong> " + selectedNode.label + 
                                   "<br><strong>接続されているノード:</strong> " + connectedNodeLabels.join(", ");
            });

            // Filter edges function
            window.filterEdges = function(color) {
              var edges = network.body.data.edges.get();
              edges.forEach(function(edge) {
                edge.hidden = (color !== "all" && edge.color !== color);
              });
              network.body.data.edges.update(edges);
            };
          }, 1000);  // 1-second delay to ensure network is ready
        });
    </script>

    <div id="node-info" style="margin-bottom: 20px; font-size: 16px; color: black;">
      <!-- ノード情報はここに表示されます -->
      ノードを選択してください。
    </div>
    """
    
    # HTMLを直接編集
    with open(output_file_path5, "r", encoding="utf-8") as file:
        html_content = file.read()

    # カスタムHTMLを挿入（ボタンを<body>タグ直後に追加）
    html_content = html_content.replace("<body>", f"<body>{custom_html}")

    # 修正後のHTMLを保存
    with open(output_file_path5, "w", encoding="utf-8") as file:
        file.write(html_content)

    print(f"ネットワークを {output_file_path5} に保存しました!!!!!!")

    f1.close() 
    f2.close() 




