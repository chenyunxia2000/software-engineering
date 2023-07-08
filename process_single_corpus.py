import pickle
from collections import Counter
import json
from collections import defaultdict
import json
import pickle
from collections import Counter, defaultdict

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def single_list(arr, target):
    return arr.count(target)

def split_data_by_qid(total_data):
    """
    根据每个问题 ID 的候选项数将数据拆分为两个列表。
    """
    qid_count = defaultdict(int)
    for data in total_data:
        qid_count[data[0][0]] += 1
    total_data_single = [data for data in total_data if qid_count[data[0][0]] == 1]
    total_data_multiple = [data for data in total_data if qid_count[data[0][0]] > 1]
    return total_data_single, total_data_multiple

def process_data_staqc(filepath, save_single_path, save_multiple_path):
    """
    通过根据每个问题ID的候选项数将数据拆分为两个列表来处理STS-B Q&A数据集。
    """
    with open(filepath, 'r') as f:
        total_data = json.load(f)
    total_data_single, total_data_multiple = split_data_by_qid(total_data)
    with open(save_single_path, 'w') as f:
        json.dump(total_data_single, f)
    with open(save_multiple_path, 'w') as f:
        json.dump(total_data_multiple, f)

def process_data_large(filepath, save_single_path, save_multiple_path):
    """
    通过根据每个问题ID的候选数量将数据拆分为两个列表来处理大规模STS-B问答数据集。
    """
    total_data = load_pickle(filepath)
    qids = [data[0][0] for data in total_data]
    result = Counter(qids)
    total_data_single = [data for data in total_data if result[data[0][0]] == 1]
    total_data_multiple = [data for data in total_data if result[data[0][0]] > 1]
    with open(save_single_path, 'wb') as f:
        pickle.dump(total_data_single, f)
    with open(save_multiple_path, 'wb') as f:
        pickle.dump(total_data_multiple, f)

def convert_single_unlabeled_to_labeled(path1, path2):
    """
    通过为每个问题 ID 添加标签 1，将单个候选数据转换为标记数据。
    """
    total_data = load_pickle(path1)
    labels = [[data[0], 1] for data in total_data]
    total_data_sort = sorted(labels, key=lambda x: (x[0], x[1]))
    with open(path2, "w") as f:
        json.dump(total_data_sort, f)

# example usage
process_data_staqc('sts-b/train-staqc.json', 'sts-b/train-staqc-single.json', 'sts-b/train-staqc-multiple.json')
process_data_large('sts-b-large/train.pickle', 'sts-b-large/train-single.pickle', 'sts-b-large/train-multiple.pickle')
convert_single_unlabeled_to_labeled('sts-b-large/train-single.pickle', 'sts-b-large/train-single-labeled.json')

if __name__ == "__main__":
    #将staqc_python中的单候选和多候选分开
    staqc_python_path = 'C:/Users/chenyunxia/Desktop/codecs/data_processing/hnn_process/ulabel_data/python_staqc_qid2index_blocks_unlabeled.txt'
    staqc_python_sigle_save ='C:/Users/chenyunxia/Desktop/codecs/data_processing/hnn_process/ulabel_data/staqc/single/python_staqc_single.txt'
    staqc_python_multiple_save = 'C:/Users/chenyunxia/Desktop/codecs/data_processing/hnn_process/ulabel_data/staqc/multiple/python_staqc_multiple.txt'
    #data_staqc_prpcessing(staqc_python_path,staqc_python_sigle_save,staqc_python_multiple_save)

    #将staqc_sql中的单候选和多候选分开
    staqc_sql_path = 'C:/Users/chenyunxia/Desktop/codecs/data_processing/hnn_process/ulabel_data/sql_staqc_qid2index_blocks_unlabeled.txt'
    staqc_sql_sigle_save = 'C:/Users/chenyunxia/Desktop/codecs/data_processing/hnn_process/ulabel_data/staqc/single/sql_staqc_single.txt'
    staqc_sql_multiple_save = 'C:/Users/chenyunxia/Desktop/codecs/data_processing/hnn_process/ulabel_data/staqc/multiple/sql_staqc_multiple.txt'
    #data_staqc_prpcessing(staqc_sql_path, staqc_sql_sigle_save, staqc_sql_multiple_save)
