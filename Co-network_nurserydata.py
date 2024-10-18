import pandas as pd
import MeCab
import time

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
    cooc_net = Network(height="750px", width="100%", bgcolor="#FFFFFF", font_color="black")

    # データフレームの作成
    data = pd.DataFrame(most_common_combinations, columns=['word_pair', 'count'])
    data[['word1', 'word2']] = pd.DataFrame(data['word_pair'].tolist(), index=data.index)

    # エッジデータの追加
    for index, row in data.iterrows():
        cooc_net.add_node(row['word1'], label=row['word1'], title=row['word1'])
        cooc_net.add_node(row['word2'], label=row['word2'], title=row['word2'])
        cooc_net.add_edge(row['word1'], row['word2'], value=row['count'])

    # ネットワークの可視化設定
    cooc_net.set_options("""
    var options = {
      "nodes": {
        "font": {
          "size": 12
        }
      },
      "edges": {
        "color": {
          "inherit": true
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
        "maxVelocity": 146,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {
          "enabled": true,
          "iterations": 2000,
          "updateInterval": 25
        }
      }
    }
    """)

    return cooc_net

# この関数を使用するには、most_common_combinationsを引数として渡します
input_file_path = 'meishi_example.txt'  # 入力ファイルのパスを設定してください
most_common_combinations = count_noun_combinations(input_file_path)
cooccurrence_network = create_cooccurrence_network(most_common_combinations)
cooccurrence_network.show("cooccurrence_network.html")

from IPython.display import HTML
HTML("/content/cooccurrence_network.html")


