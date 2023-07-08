import pickle

def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def build_vocab(corpus1, corpus2):
    """
    构建初始词典
    :param corpus1: 数据集1
    :param corpus2: 数据集2
    :return: 词典
    """
    vocab = set()
    for corpus in [corpus1, corpus2]:
        for i in range(len(corpus)):
            # 添加所有出现过的单词
            vocab.update(corpus[i][1][0])
            vocab.update(corpus[i][1][1])
            vocab.update(corpus[i][2][0])
            vocab.update(corpus[i][3])
    return vocab


def save_vocab(vocab, save_path):
    """
    保存词典到文件
    :param vocab: 词典
    :param save_path: 保存路径
    """
    with open(save_path, "w") as f:
        f.write(str(vocab))


def process_vocab_file(vocab_file_path, filter_file_path, output_file_path):
    """
    根据已有词典和过滤文件，生成最终的词典
    :param vocab_file_path: 已有词典文件路径
    :param filter_file_path: 过滤文件路径
    :param output_file_path: 生成词典文件路径
    """
    # 加载已有词典和过滤文件
    vocab = load_pickle(vocab_file_path)
    with open(filter_file_path, 'r') as f:
        filter_words = set(eval(f.read()))
    # 去除过滤文件中的词
    vocab = vocab - filter_words
    # 保存最终词典到文件
    save_vocab(vocab, output_file_path)


# 测试代码
if __name__ == '__main__':
    corpus1 = load_pickle('corpus1.pkl')
    corpus2 = load_pickle('corpus2.pkl')
    vocab = build_vocab(corpus1, corpus2)
    save_vocab(vocab, 'vocab_initial.txt')
    process_vocab_file('vocab_initial.txt', 'filter_words.txt', 'vocab_final.txt')
'''
if __name__ == "__main__":
    #====================获取staqc的词语集合===============
    python_hnn = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/python_hnn_data_teacher.txt'
    python_staqc = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/staqc/python_staqc_data.txt'
    python_word_dict = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/word_dict/python_word_vocab_dict.txt'

    sql_hnn = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/sql_hnn_data_teacher.txt'
    sql_staqc = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/staqc/sql_staqc_data.txt'
    sql_word_dict = '/home/gpu/RenQ/stqac/data_processing/hnn_process/data/word_dict/sql_word_vocab_dict.txt'

    # vocab_prpcessing(python_hnn,python_staqc,python_word_dict)
    # vocab_prpcessing(sql_hnn,sql_staqc,sql_word_dict)
    #====================获取最后大语料的词语集合的词语集合===============
    new_sql_staqc = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabled_data.txt'
    new_sql_large = '../hnn_process/ulabel_data/large_corpus/multiple/sql_large_multiple_unlable.txt'
    large_word_dict_sql = '../hnn_process/ulabel_data/sql_word_dict.txt'
    final_vocab_prpcessing(sql_word_dict, new_sql_large, large_word_dict_sql)
    #vocab_prpcessing(new_sql_staqc,new_sql_large,final_word_dict_sql)

    new_python_staqc = '../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.txt'
    new_python_large ='../hnn_process/ulabel_data/large_corpus/multiple/python_large_multiple_unlable.txt'
    large_word_dict_python = '../hnn_process/ulabel_data/python_word_dict.txt'
    #final_vocab_prpcessing(python_word_dict, new_python_large, large_word_dict_python)
    #vocab_prpcessing(new_python_staqc,new_python_large,final_word_dict_python)
'''



