import os
import itertools
import collections
from pyvis.network import Network
import traceback
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# idfをすべての保育所の単語リストとした時
def compute_tfidf(all_word_lists, target_word_list):
    """
    - all_word_lists: すべての保育所の単語リスト（リストのリスト）
    - target_word_list: 特定の保育所の単語リスト
    """
   
    # TF-IDF 計算
    vecs0 = TfidfVectorizer(token_pattern=u'(?u)\\b\\w+\\b')
    vecs = vecs0.fit_transform(all_word_lists)
    
    # 全体の単語リスト
    terms = vecs0.get_feature_names_out()

    # 特定の保育所データに対するTF-IDFベクトルを計算
    target_doc = ' '.join(target_word_list)
    target_vec = vecs0.transform([target_doc]).toarray()[0]

    # TF-IDF スコアの高い単語を取得
    top_n_idx = target_vec.argsort()[-100:][::-1]  # 上位100単語
    top_words = [terms[idx] for idx in top_n_idx]
    return top_words

def count_noun_combinations(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        sentences = [line.strip().split() for line in file if line.strip()]
    sentences_combs = [list(itertools.combinations(sentence, 2)) for sentence in sentences]
    #all_combinations = [tuple(sorted(comb)) for sentence in sentences_combs for comb in sentence]
    #counter = collections.Counter(all_combinations)
    #return counter.most_common(top_n)
    sorted_combs = [[tuple(sorted(comb)) for comb in sentence] for sentence in sentences_combs]

    target_combs = []
    for words_comb in sorted_combs:
        target_combs.extend(words_comb)

    # 単語の組み合わせをカウント
    ct = collections.Counter(target_combs)
    return ct
 
def create_all_network_combinations(directory_path):
    # .txtファイルを取得
    txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    
    # 全ファイルの組み合わせを生成
    file_combinations = list(itertools.combinations(txt_files, 2))
    
    for file1, file2 in file_combinations:
        try:
            file1_path = os.path.join(directory_path, file1)
            file2_path = os.path.join(directory_path, file2)
            
            # ファイル名から保育園名を抽出
            nursery1_name = re.sub(r'^meishi_example_|\.txt$', '', file1)
            nursery2_name = re.sub(r'^meishi_example_|\.txt$', '', file2)
            
            # 出力ファイル名
            output_path = f'./analize_nurseryschool_app/public/network_vs/network_{nursery1_name}_vs_{nursery2_name}.html'
            
            # ネットワーク分析と可視化
            visualize_two_nursery_network(file1_path, file2_path, nursery1_name, nursery2_name, output_path)
            
        except Exception as e:
            print(f"エラー: {file1} と {file2} の処理中にエラーが発生しました")
            traceback.print_exc()

# 2つのネットワークを統合して差分を可視化する関数
excluded_names = ["山内保育所（西部）", "越来保育所", "知花保育所"]
def visualize_two_nursery_network(file1_path, file2_path, nursery1_name, nursery2_name, output_path):
    # 除外対象の保育園はスキップ
    if nursery1_name in excluded_names or nursery2_name in excluded_names:
        print(f"スキップ: {nursery1_name} または {nursery2_name} は除外対象です。")
        return

    # 各ファイルから名詞ペアを抽出
    most_common_a = count_noun_combinations(file1_path)
    most_common_b = count_noun_combinations(file2_path)

    # 辞書形式に変換
    dict_a = dict(most_common_a)
    dict_b = dict(most_common_b)

    # 共通部分と差分を計算
    common_edges = set(dict_a.keys()) & set(dict_b.keys())
    unique_a_edges = set(dict_a.keys()) - set(dict_b.keys())
    unique_b_edges = set(dict_b.keys()) - set(dict_a.keys())

    # PyVisネットワークを作成
    network = Network(height="750px", width="100%", bgcolor="#FFFFFF", font_color="black")

    ## 保存したノード位置ファイルを読み込む
    with open('mds_node_positions_v3.json', 'r') as f:
        node_positions = json.load(f)
    print("the sum of keys!!!!!", len(node_positions.keys()))
    ## 位置情報が存在するノードのみをフィルタリングして描画
    all_nodes = set(itertools.chain(*common_edges, *unique_a_edges, *unique_b_edges))
    filtered_positions = {node: pos for node, pos in node_positions.items() if node in all_nodes}
    filtered_nodes = list(filtered_positions.keys())  # 位置情報があるノードのリスト
    print("filtered_nodes!!!!!", filtered_nodes)
    print("filtered_positions!!!!!", filtered_positions)

    # === TF-IDF による重要語抽出 ===
    with open(file1_path, 'r', encoding='utf-8') as f:
        word_lists_a = [line.strip() for line in f if line.strip()]
    with open(file2_path, 'r', encoding='utf-8') as f:
        word_lists_b = [line.strip() for line in f if line.strip()]

    all_word_lists = word_lists_a + word_lists_b
    target_word_list_a = [word for line in word_lists_a for word in line.split()]
    target_word_list_b = [word for line in word_lists_b for word in line.split()]

    top_words_a = compute_tfidf(all_word_lists, target_word_list_a)
    top_words_b = compute_tfidf(all_word_lists, target_word_list_b)
    important_words = set(top_words_a + top_words_b)

    filtered_nodes = [node for node in filtered_positions if node in important_words]
    for node in filtered_nodes:
        x, y = filtered_positions[node]
        network.add_node(node, x=x, y=y, fixed=True, size=5, color={"highlight": 'gray', "hover": 'gray'})
    # 保存したノード位置を使用した場合(tfidf使用せず)
    #for node in filtered_positions:
    #    x, y = filtered_positions[node]
    #    network.add_node(node, x=x, y=y, fixed=True, size=5, color={"highlight": 'gray', "hover": 'gray'})
    
    # エッジを追加（共通=緑色、差分＝赤と青）
    # TFIDFを使用
    edge_id = 0  # ← 追加
    for edge in unique_a_edges:
        if edge[0] in filtered_nodes and edge[1] in filtered_nodes:
            network.add_edge(edge[0], edge[1], value=dict_a[edge], color='#FF4B00', id=str(edge_id))
            edge_id += 1

    for edge in unique_b_edges:
        if edge[0] in filtered_nodes and edge[1] in filtered_nodes:
            network.add_edge(edge[0], edge[1], value=dict_b[edge], color='#005AFF', id=str(edge_id))
            edge_id += 1

    for edge in common_edges:
        if edge[0] in filtered_nodes and edge[1] in filtered_nodes:
            network.add_edge(edge[0], edge[1], value=dict_a[edge], color='#03AF7A', id=str(edge_id))
            edge_id += 1

    # TFIDFを使用せず
    #edge_id = 0  # ← 追加
    #for edge in unique_a_edges:
    #    if edge[0] in filtered_positions and edge[1] in filtered_positions:
    #        network.add_edge(edge[0], edge[1], value=dict_a[edge], color='#FF4B00', id=str(edge_id))
    #        edge_id += 1

    #for edge in unique_b_edges:
    #    if edge[0] in filtered_positions and edge[1] in filtered_positions:
    #        network.add_edge(edge[0], edge[1], value=dict_b[edge], color='#005AFF', id=str(edge_id))
    #        edge_id += 1

    #for edge in common_edges:
    #    if edge[0] in filtered_positions and edge[1] in filtered_positions:
    #        network.add_edge(edge[0], edge[1], value=dict_a[edge], color='#03AF7A', id=str(edge_id))
    #        edge_id += 1


    # 最大エッジ重み取得（スライダー最大値用）
    weight_list = list(most_common_a.values()) + list(most_common_b.values())
    max_weight = max(weight_list) if weight_list else 1

    network.set_options("""
    {
      "interaction": {
        "hover": true,
        "selectConnectedEdges": true
      },
      "nodes": {
        "font": {
          "size": 15 
        },
        "scaling": {
          "min": 5,
          "max": 10
        },
        "color": {
           "highlight": {
             "border": "black"
           }
        }
      },
      "edges": {
        "smooth": false
      },
      "physics": {
      "enabled": false
      }
   }
    """)
    
    # ネットワークの保存
    #network.show(output_path)
    network.write_html(output_path, notebook=False, local=False)
    print(f"統合ネットワークを {output_path} に保存しました。")

    # === ノード情報表示用UIを追加 ===
    with open(output_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    # 不要な <script src="lib/bindings/utils.js"> を削除
    html_content = html_content.replace('<script src="lib/bindings/utils.js"></script>', '')

    # === 文章データを取得する ===
    nursery1_sentence_path = './nursery_school/MeCab_EachNurserySchool/sentences_' + nursery1_name + '.txt'
    nursery2_sentence_path = './nursery_school/MeCab_EachNurserySchool/sentences_' + nursery2_name + '.txt'
    nursery1_sentences = open(nursery1_sentence_path, 'r')
    nursery2_sentences = open(nursery2_sentence_path, 'r')
    n1_sentences = nursery1_sentences.read()
    n1_sentences = n1_sentences.split('""')
    n1_sentences = n1_sentences[1:] 
    n1_sentences = [dt for dt in n1_sentences if dt != '""' and dt != '"']
    n2_sentences = nursery2_sentences.read()
    n2_sentences = n2_sentences.split('""')
    n2_sentences = n2_sentences[1:] 
    n2_sentences = [dt for dt in n2_sentences if dt != '""' and dt != '"']
    sentences = n1_sentences + n2_sentences
    
    #sentence_js = f"var sentenceData = {json.dumps(sentences, ensure_ascii=False)};"
    sentence_js = f"""
      var sentenceDataA = {json.dumps(n1_sentences, ensure_ascii=False)};
      var sentenceDataB = {json.dumps(n2_sentences, ensure_ascii=False)};
      var sentenceData = sentenceDataA.concat(sentenceDataB);
    """
            
    # 移動用: ネットワーク描画部分を抽出して後で移動
    start_div = html_content.find('<div id="mynetwork"')
    end_div = html_content.find('</script>', start_div) + len('</script>')
    mynetwork_block = html_content[start_div:end_div]

    info_div = f"""
    <div style="margin-bottom: 10px;">
      <label for="weightSlider">共起回数（エッジの重み）以上を表示: </label>
      <input type="range" id="weightSlider" min="1" max="{max_weight}" value="1">
      <span id="weightValue">1</span>
    </div>
    <div id=\"nodeInfo\" style=\"margin-bottom: 20px; font-size: 16px; font-weight: bold;\">選択したノードの情報がここに表示されます</div>
    <div style=\"margin-bottom: 10px;\">
      <button onclick=\"filterEdgesByColor('all')\">すべて表示</button>
      <button onclick=\"filterEdgesByColor('#03AF7A')\">共通（緑）</button>
      <button onclick=\"filterEdgesByColor('#FF4B00')\">nurseryA（赤）</button>
      <button onclick=\"filterEdgesByColor('#005AFF')\">nurseryB（青）</button>
    </div>
    <div id="container" style="display: flex; flex-direction: row;">
      <div id="relatedSentences" style="width: 30%; padding: 10px; border: 1px solid #ccc; background: #f9f9f9; font-size: 14px; overflow-y: auto; height: 750px;">
        <strong>関連文:</strong><br>
      </div>
      <div style="width: 70%; padding-left: 10px;">
        {mynetwork_block}
      </div>
    </div>
    <script>{sentence_js}</script>
    """

    # <body>〜</body> の中身を書き換える
    start_body = html_content.find('<body>') + len('<body>')
    end_body = html_content.find('</body>')
    new_html = html_content[:start_body] + info_div + html_content[end_body:]

    script_code = """
    <script type=\"text/javascript\">
      function filterEdgesByColor(color) {
        const edgesDataSet = network.body.data.edges;
        const allEdges = edgesDataSet.get();
        const updatedEdges = allEdges.map(edge => {
          if (color === 'all') {
            return { id: edge.id, hidden: false };
          } else {
            return { id: edge.id, hidden: edge.color !== color };
          }
        });

        edgesDataSet.update(updatedEdges);  // ← ここでまとめて更新        
     }

      document.addEventListener("DOMContentLoaded", function() {
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

        function updateNetwork(minWeight) {
            var filteredEdges = originalEdges.filter(e => e.value >= minWeight);
            var usedNodeIds = new Set();
            filteredEdges.forEach(e => {
                usedNodeIds.add(e.from);
                usedNodeIds.add(e.to);
            });
            var filteredNodes = originalNodes.filter(n => usedNodeIds.has(n.id));
            var data = { nodes: new vis.DataSet(filteredNodes), edges: new vis.DataSet(filteredEdges) };
            network.setData(data);
        }
  
        // ノードクリック時の情報表示
        network.on("click", function(params) {
            if (params.nodes.length > 0) {
                var selectedNode = params.nodes[0];
                var connectedNodes = network.getConnectedNodes(selectedNode);
                var text = "選択ノード: " + selectedNode + "<br>接続ノード: " + connectedNodes.join(", ");
                document.getElementById("nodeInfo").innerHTML = text;

                var related = sentenceData.filter(s => s.includes(selectedNode));
                //var highlighted = related.map(s => s.replaceAll(selectedNode, `<span style=\"color:red\">${selectedNode}</span>`));
                var highlighted = related.map(s => {
                  let color = sentenceDataA.includes(s) ? "#FF4B00" : (sentenceDataB.includes(s) ? "#005AFF" : "black");
                  return s.replaceAll(selectedNode, `<span style="color:${color}">${selectedNode}</span>`);
                });
                document.getElementById("relatedSentences").innerHTML = "<strong>関連文:</strong><br>" + highlighted.join("<br><br>");
            }
        });
      });
      </script>
    """

    updated_html = new_html.replace("</body>", script_code + "\n</body>")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(updated_html)

    print(f"ネットワークを {output_path} に保存しました。")


# ディレクトリパス
directory_path = './analize_nurseryschool_app/public/Noun_EachNurserySchool/'

# 全保育園のネットワークを分析
create_all_network_combinations(directory_path)
