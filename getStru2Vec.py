import os
import pickle
import logging
import sys
sys.path.append("..")
from multiprocessing.pool import ThreadPool

from typing import List
from multiprocessing.dummy import Pool as ThreadPool

# 解析结构
from python_structured import *
from sqlang_structured import *

#FastText库  gensim 3.4.0
from gensim.models import FastText

import numpy as np

#词频统计库
import collections
#词云展示库
import wordcloud
#图像处理库 Pillow 5.1.0
from PIL import Image

# 多进程
from multiprocessing import Pool as ThreadPool


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def multiprocess_data(data_list: List[str], process_func) -> List:
    """
    使用多进程并行处理数据列表
    :param data_list: 需要处理的数据列表
    :param process_func: 处理函数，接受一个数据列表作为输入，返回一个处理结果列表
    :return: 处理结果列表
    """
    try:
        split_num = 1000  # 每个进程处理的数据量
        split_list = [data_list[i:i + split_num] for i in range(0, len(data_list), split_num)]
        pool = ThreadPool(10)
        result_list = pool.map(process_func, split_list)
        pool.close()
        pool.join()
        result = [p for sublist in result_list for p in sublist]
        logging.info('处理数据条数：%d' % len(result))
        return result
    except Exception as e:
        logging.error('数据处理出错：%s' % e)
        return []


# python解析
def multiprocess_python_query(data_list: List[str]) -> List:
    # 处理 query 的函数
    pass


def multiprocess_python_code(data_list: List[str]) -> List:
    # 处理 code 的函数
    pass


def multiprocess_python_context(data_list: List[str]) -> List:
    # 处理 acont1 和 acont2 的函数
    pass


# sql解析
def multiprocess_sqlang_query(data_list: List[str]) -> List:
    # 处理 query 的函数
    pass


def multiprocess_sqlang_code(data_list: List[str]) -> List:
    # 处理 code 的函数
    pass


def multiprocess_sqlang_context(data_list: List[str]) -> List:
    # 处理 acont1 和 acont2 的函数
    pass


# parser.py
import logging
import pickle
from typing import List, Tuple
from utils import *


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_python(python_list: List[Tuple], split_num: int) -> Tuple[List, List, List, List, List]:
    """
    解析 Python 代码，将其分为 acont1、acont2、query 和 code 四个部分
    :param python_list: 待解析的 Python 代码列表
    :param split_num: 每个进程处理的数据量
    :return: acont1、acont2、query、code 和 qids 五个部分的列表
    """
    try:
        acont1_data = [i[1][0][0] for i in python_list]
        acont1_cut = multiprocess_data(acont1_data, multiprocess_python_context)

        acont2_data = [i[1][1][0] for i in python_list]
        acont2_cut = multiprocess_data(acont2_data, multiprocess_python_context)

        query_data = [i[3][0] for i in python_list]
        query_cut = multiprocess_data(query_data, multiprocess_python_query)

        code_data = [i[2][0][0] for i in python_list]
        code_cut = multiprocess_data(code_data, multiprocess_python_code)

        qids = [i[0] for i in python_list]
        logging.info('解析 Python 代码完成，共解析 %d 条数据' % len(qids))

        return acont1_cut, acont2_cut, query_cut, code_cut, qids
    except Exception as e:
        logging.error('解析 Python 代码出错：%s' % e)
        return [], [], [], [], []


def parse_sqlang(sqlang_list:List[Tuple], split_num: int) -> Tuple[List, List, List, List, List]:
    """
    解析 SQL 代码，将其分为 acont1、acont2、query 和 code 四个部分
    :param sqlang_list: 待解析的 SQL 代码列表
    :param split_num: 每个进程处理的数据量
    :return: acont1、acont2、query、code 和 qids 五个部分的列表
    """
    try:
        acont1_data = [i[1][0][0] for i in sqlang_list]
        acont1_cut = multiprocess_data(acont1_data, multiprocess_sqlang_context)

        acont2_data = [i[1][1][0] for i in sqlang_list]
        acont2_cut = multiprocess_data(acont2_data, multiprocess_sqlang_context)

        query_data = [i[3][0] for i in sqlang_list]
        query_cut = multiprocess_data(query_data, multiprocess_sqlang_query)

        code_data = [i[2][0][0] for i in sqlang_list]
        code_cut = multiprocess_data(code_data, multiprocess_sqlang_code)

        qids = [i[0] for i in sqlang_list]
        logging.info('解析 SQL 代码完成，共解析 %d 条数据' % len(qids))

        return acont1_cut, acont2_cut, query_cut, code_cut, qids
    except Exception as e:
        logging.error('解析 SQL 代码出错：%s' % e)
        return [], [], [], [], []


def parse_data(data_file: str, output_file: str):
    """
    解析数据文件中的 Python 和 SQL 代码，并将解析结果保存到输出文件中
    :param data_file: 待解析的数据文件
    :param output_file: 解析结果输出文件
    """
    try:
        with open(data_file, 'rb') as f:
            data = pickle.load(f)

        python_data = [(i['id'], i['content'], i['code'], i['query']) for i in data if i['type'] == 'python']
        python_cut = parse_python(python_data, 1000)

        sqlang_data = [(i['id'], i['content'], i['code'], i['query']) for i in data if i['type'] == 'sql']
        sqlang_cut = parse_sqlang(sqlang_data, 1000)

        result = {'python': python_cut, 'sqlang': sqlang_cut}
        with open(output_file, 'wb') as f:
            pickle.dump(result, f)

        logging.info('解析数据文件完成，结果已保存到 %s' % output_file)
    except Exception as e:
        logging.error('解析数据文件出错：%s' % e)

if __name__ == '__main__':
    lang_type = 'python'
    split_num = 100
    source_path = 'data.pkl'
    save_path = 'total_data.pkl'
    main(lang_type, split_num, source_path, save_path)


python_type= 'python'
sqlang_type ='sql'

words_top = 100

split_num = 1000
def test(path1,path2):
    with open(path1, "rb") as f:
        #  存储为字典 有序
        corpus_lis1  = pickle.load(f) #pickle
    with open(path2, "rb") as f:
        corpus_lis2 = eval(f.read()) #txt

    print(corpus_lis1[10])
    print(corpus_lis2[10])


if __name__ == '__main__':
    staqc_python_path = '../hnn_process/ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_save ='../hnn_process/ulabel_data/staqc/python_staqc_unlabled_data.txt'

    staqc_sql_path = '../hnn_process/ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_save = '../hnn_process/ulabel_data/staqc/sql_staqc_unlabled_data.txt'

    #main(sqlang_type,split_num,staqc_sql_path,staqc_sql_save)
    #main(python_type, split_num, staqc_python_path, staqc_python_save)
