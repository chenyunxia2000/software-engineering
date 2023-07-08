
'''

从大词典中获取特定于于语料的词典
将数据处理成待打标签的形式
'''

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from gensim.models import KeyedVectors
import os
import numpy as np
from gensim.models import KeyedVectors


EMBEDDING_DIM = 300


def save_word2vec_as_bin(word2vec_path, bin_path):
    """
    将词向量文件保存成二进制文件，以提高加载速度。

    Args:
        word2vec_path: str, 词向量文件的路径。
        bin_path: str, 保存二进制文件的路径。
    """
    if not os.path.isfile(word2vec_path):
        print(f"{word2vec_path} 文件不存在。")
        return

    word_vectors = KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
    word_vectors.init_sims(replace=True)
    word_vectors.save(bin_path)


def build_new_dict(path_to_word2vec, path_to_word_list, path_to_embedding, path_to_word_index):
    """
    根据词向量文件和词汇表，构建新的词典和词向量矩阵。

    Args:
        path_to_word2vec: str, 词向量文件的路径。
        path_to_word_list: str, 词汇表的路径。
        path_to_embedding: str, 保存词向量矩阵的路径。
        path_to_word_index: str, 保存词典的路径。
    """
    if not os.path.isfile(path_to_word2vec):
        print(f"{path_to_word2vec} 文件不存在。")
        return
    if not os.path.isfile(path_to_word_list):
        print(f"{path_to_word_list} 文件不存在。")
        return

    with open(path_to_word_list, 'r') as f:
        word_list = eval(f.read())

    word_index = {'PAD': 0, 'SOS': 1, 'EOS': 2, 'UNK': 3}
    embedding_matrix = np.zeros((4, EMBEDDING_DIM))

    # 初始化特殊词的词向量
    rng = np.random.RandomState(None)
    embedding_matrix[word_index['PAD']] = np.zeros((EMBEDDING_DIM,))
    embedding_matrix[word_index['SOS']] = rng.uniform(-0.25, 0.25, size=(EMBEDDING_DIM,))
    embedding_matrix[word_index['EOS']] = rng.uniform(-0.25, 0.25, size=(EMBEDDING_DIM,))
    embedding_matrix[word_index['UNK']] = rng.uniform(-0.25, 0.25, size=(EMBEDDING_DIM,))

    # 加载词向量文件，构建新的词典和词向量矩阵
    try:
        word_vectors = KeyedVectors.load(path_to_word2vec, mmap='r')
    except Exception as e:
        print(f"加载词向量文件失败：{e}")
        return

    for word in word_list:
        if word not in word_index:
            word_index[word] = len(word_index)
        try:
            embedding_matrix = np.vstack((embedding_matrix, word_vectors[word]))
        except KeyError:
            print(f"未能找到词向量：{word}")
            embedding_matrix = np.vstack((embedding_matrix, rng.uniform(-0.25, 0.25, size=(EMBEDDING_DIM,))))

    # 保存词向量矩阵和词典
    os.makedirs(os.path.dirname(path_to_embedding), exist_ok=True)
    os.makedirs(os.path.dirname(path_to_word_index), exist_ok=True)

    np.save(path_to_embedding, embedding_matrix)
    np.save(path_to_word_index, word_index)

    print("完成")


VOCAB_SIZE = 10000
MAX_QUERY_LEN = 25
MAX_SENTENCE_LEN = 100
MAX_CODE_LEN = 350


class TextType:
    CODE = 'code'
    TEXT = 'text'


def get_index(text_type, text, word_index):
    """获取文本在词典中的位置。

    Args:
        text_type: TextType, 文本类型，枚举类型。
        text: list of str, 文本列表。
        word_index: dict, 词典，从单词到索引的映射。

    Returns:
        location: list of int, 文本在词典中的位置列表。
    """
    location = []
    if text_type == TextType.CODE:
        location.append(1)
        len_c = len(text)
        if len_c + 1 < MAX_CODE_LEN:
            if len_c == 1 and text[0] == '-1000':
                location.append(2)
            else:
                for i in range(len_c):
                    index = word_index.get(text[i], word_index['UNK'])
                    location.append(index)
                location.append(2)
        else:
            for i in range(MAX_CODE_LEN - 2):
                index = word_index.get(text[i], word_index['UNK'])
                location.append(index)
            location.append(2)
    else:
        if len(text) == 0:
            location.append(0)
        elif text[0] == '-10000':
            location.append(0)
        else:
            for i in range(len(text)):
                index = word_index.get(text[i], word_index['UNK'])
                location.append(index)
    return location


def serialize_data(path_to_word_dict, path_to_input_file, path_to_output_file):
    """将训练、测试或验证语料序列化。

    Args:
        path_to_word_dict: str, 词典文件的路径。
        path_to_input_file: str, 输入文件的路径。
        path_to_output_file: str, 输出文件的路径。
    """
    if not os.path.isfile(path_to_word_dict):
        print(f"{path_to_word_dict} 文件不存在。")
        return
    if not os.path.isfile(path_to_input_file):
        print(f"{path_to_input_file} 文件不存在。")
        return

    with open(path_to_word_dict, 'rb') as f:
        word_index = np.load(f, allow_pickle=True).item()

    with open(path_to_input_file, 'r') as f:
        corpus = eval(f.read())

    total_data = []

    for i in range(len(corpus)):
        qid = corpus[i][0]
        prev_sentence = get_index(TextType.TEXT, corpus[i][1][0], word_index)
        next_sentence = get_index(TextType.TEXT, corpus[i][1][1], word_index)
       code_tokens = get_index(TextType.CODE, corpus[i][2][0], word_index)
        query = get_index(TextType.TEXT, corpus[i][3], word_index)
        block_length = 4
        label = 0

        prev_sentence = prev_sentence[:MAX_SENTENCE_LEN] + [0] * (MAX_SENTENCE_LEN - len(prev_sentence))
        next_sentence = next_sentence[:MAX_SENTENCE_LEN] + [0] * (MAX_SENTENCE_LEN - len(next_sentence))
        code_tokens = code_tokens[:MAX_CODE_LEN] + [0] * (MAX_CODE_LEN - len(code_tokens))
        query = query[:MAX_QUERY_LEN] + [0] * (MAX_QUERY_LEN - len(query))

        one_data = {
            'qid': qid,
            'prev_sentence': prev_sentence,
            'next_sentence': next_sentence,
            'code_tokens': code_tokens,
            'query': query,
            'block_length': block_length,
            'label': label,
        }
        total_data.append(one_data)

    with open(path_to_output_file, 'wb') as f:
      np.save(f, total_data)

import os
import numpy as np
from gensim.models import KeyedVectors

VOCAB_SIZE = 159019
FAIL_WORD_COUNT = 25059


def get_new_dict_append(path_to_type_vectors, path_to_previous_dict, path_to_previous_vectors,
                        path_to_append_word, final_vec_path, final_word_path):
    """将新的词语和对应的词向量添加到原有的词典和词向量中，并将结果保存为文件。

    Args:
        path_to_type_vectors: str, 原有的词向量文件的路径。
        path_to_previous_dict: str, 原有的词典文件的路径。
        path_to_previous_vectors: str, 原有的词向量文件的路径。
        path_to_append_word: str, 要添加的词语列表文件的路径。
        final_vec_path: str, 最终的词向量文件的路径。
        final_word_path: str, 最终的词典文件的路径。
    """
    # 加载原有的词向量文件和词典文件
    model = KeyedVectors.load(path_to_type_vectors, mmap='r')
    with open(path_to_previous_dict, 'rb') as f:
        previous_word_dict = np.load(f, allow_pickle=True).item()
    with open(path_to_previous_vectors, 'rb') as f:
        previous_word_vectors = np.load(f, allow_pickle=True)

    # 加载要添加的词语列表
    with open(path_to_append_word, 'r') as f:
        append_word = eval(f.read())

    # 将新的词语和对应的词向量添加到原有的词典和词向量中
    word_dict = list(previous_word_dict.keys())
    word_vectors = previous_word_vectors.tolist()
    fail_word = []
    unk_embeddings = np.random.RandomState(None).uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    for word in append_word:
        try:
            word_vectors.append(model.wv[word])
            word_dict.append(word)
        except:
            fail_word.append(word)

    # 输出结果
    print(f"原有词典大小：{len(previous_word_dict)}，添加的词语数目：{len(append_word)}，"
          f"最终词典大小：{len(word_dict)}，添加失败的词语数目：{len(fail_word)}")
    print(f"原有词向量大小：{previous_word_vectors.shape}，添加的词向量大小：{len(append_word)}，"
          f"最终词向量大小：{np.array(word_vectors).shape}")

    # 将结果保存为文件
    word_vectors = np.array(word_vectors)
    word_dict = dict(map(reversed, enumerate(word_dict)))
    with open(final_vec_path, 'wb') as f:
        np.save(f, word_vectors)
    with open(final_word_path, 'wb') as f:
        np.save(f, word_dict)
    print("保存成功")


import time

#-------------------------参数配置----------------------------------
#python 词典 ：1121543 300
if __name__ == '__main__':

    ps_path = '../hnn_process/embeddings/10_10/python_struc2vec1/data/python_struc2vec.txt' #239s
    ps_path_bin = '../hnn_process/embeddings/10_10/python_struc2vec.bin' #2s

    sql_path = '../hnn_process/embeddings/10_8_embeddings/sql_struc2vec.txt'
    sql_path_bin = '../hnn_process/embeddings/10_8_embeddings/sql_struc2vec.bin'

    #trans_bin(sql_path,sql_path_bin)
    #trans_bin(ps_path, ps_path_bin)
    #113440 27970(2) 49409(12),50226(30),55993(98)

    #==========================  ==========最初基于Staqc的词典和词向量==========================

    python_word_path = '../hnn_process/data/word_dict/python_word_vocab_dict.txt'
    python_word_vec_path = '../hnn_process/embeddings/python/python_word_vocab_final.pkl'
    python_word_dict_path = '../hnn_process/embeddings/python/python_word_dict_final.pkl'

    sql_word_path = '../hnn_process/data/word_dict/sql_word_vocab_dict.txt'
    sql_word_vec_path = '../hnn_process/embeddings/sql/sql_word_vocab_final.pkl'
    sql_word_dict_path = '../hnn_process/embeddings/sql/sql_word_dict_final.pkl'



    #txt存储数组向量，读取时间：30s,以pickle文件存储0.23s,所以最后采用pkl文件

    #get_new_dict(ps_path_bin,python_word_path,python_word_vec_path,python_word_dict_path)
    #get_new_dict(sql_path_bin, sql_word_path, sql_word_vec_path, sql_word_dict_path)

    # =======================================最后打标签的语料========================================
    #sql 待处理语料地址
    new_sql_staqc = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabled_data.txt'
    new_sql_large = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    #sql大语料最后的词典
    large_word_dict_sql = '../hnn_process/ulabel_data/sql_word_dict.txt'

    # sql最后的词典和对应的词向量
    sql_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/sql_word_vocab_final.pkl'
    sql_final_word_dict_path = '../hnn_process/ulabel_data/large_corpus/sql_word_dict_final.pkl'
    #get_new_dict(sql_path_bin, final_word_dict_sql, sql_final_word_vec_path, sql_final_word_dict_path)
    #get_new_dict_append(sql_path_bin, sql_word_dict_path, sql_word_vec_path, large_word_dict_sql, sql_final_word_vec_path,sql_final_word_dict_path)

    staqc_sql_f = '../hnn_process/ulabel_data/staqc/seri_sql_staqc_unlabled_data.pkl'
    large_sql_f = '../hnn_process/ulabel_data/large_corpus/multiple/seri_ql_large_multiple_unlable.pkl'
    #Serialization(sql_final_word_dict_path, new_sql_staqc, staqc_sql_f)
    #Serialization(sql_final_word_dict_path, new_sql_large, large_sql_f)



    #python
    new_python_staqc = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.txt'
    new_python_large = '../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.txt'
    final_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'

    #python最后的词典和对应的词向量
    python_final_word_vec_path = '../hnn_process/ulabel_data/large_corpus/python_word_vocab_final.pkl'
    python_final_word_dict_path = '../hnn_process/ulabel_data/large_corpus/python_word_dict_final.pkl'

    #get_new_dict(ps_path_bin, final_word_dict_python, python_final_word_vec_path, python_final_word_dict_path)
    #get_new_dict_append(ps_path_bin, python_word_dict_path, python_word_vec_path, large_word_dict_python, python_final_word_vec_path,python_final_word_dict_path)

    #处理成打标签的形式
    staqc_python_f ='../hnn_process/ulabel_data/staqc/seri_python_staqc_unlabled_data.pkl'

    #Serialization(python_final_word_dict_path, new_python_staqc, staqc_python_f)
    Serialization(python_final_word_dict_path, new_python_large, large_python_f)

    print('序列化完毕')
    #test2(test_python1,test_python2,python_final_word_dict_path,python_final_word_vec_path)








