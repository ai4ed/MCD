import os
import json
import pandas as pd
import numpy as np

def load_data(data_dir,fname="df_add_features.csv"):
    df = pd.read_csv(os.path.join(data_dir, fname))
    # 读取划分的id
    train_dev_test = json.load(
        open(os.path.join(data_dir, 'train_dev_test.json'), 'r'))

    # split_data
    df_train = df[df['new_id'].isin(train_dev_test['train'])]
    df_dev = df[df['new_id'].isin(train_dev_test['dev'])]
    df_test = df[df['new_id'].isin(train_dev_test['test'])]
    print("train num {}\ndev num {}\ntest num {}".format(
        len(df_train), len(df_dev), len(df_test)))
    return df_train, df_dev, df_test

def load_save_vectors(data_dir,remove_x=False):
    print("Start load data form {}".format(data_dir))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_dev = np.load(os.path.join(data_dir, 'y_dev.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    if remove_x:#for wide models
        x_train = np.zeros((y_train.size,1,1))
        x_dev = np.zeros((y_dev.size,1,1))
        x_test = np.zeros((y_test.size,1,1))
    else:
        x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
        x_dev = np.load(os.path.join(data_dir, 'x_dev.npy'))
        x_test = np.load(os.path.join(data_dir, 'x_test.npy'))
    print("Finish load data")
    return x_train, y_train, x_dev, y_dev, x_test, y_test

def set_seed(seed):
    try:
        import tensorflow as tf
        tf.random.set_random_seed(seed)
    except Exception as e:
        print("Set seed failed,details are ", e)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as e:
        print("Set seed failed,details are ", e)
        pass
    import numpy as np
    np.random.seed(seed)
    import random as python_random
    python_random.seed(seed)
    # cuda env
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ['PYTHONHASHSEED'] = str(seed)
    
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def get_lbe_dict(df,features):
    lbe_dict = {}
    for feat in features:
        df[feat] = df[feat].apply(str)
        lbe = LabelEncoder()
        lbe.fit(df[feat])
        lbe_dict[feat] = lbe
    return lbe_dict

def load_wide_feature(data_dir, fname, config={}):
    '''load wide feature'''
    train_data, dev_data, test_data = load_data(data_dir, fname)
    # 特征的列表
    if "feature_type_file" not in config:
        feature_type_file = "feature_types"
    else:
        feature_type_file = config.feature_type_file
        
    df_feature_types = pd.read_csv(os.path.join(data_dir, '{}.csv'.format(feature_type_file)))
    data_merge = pd.concat([train_data, dev_data, test_data])
    print(data_merge.shape)

    limit_feature_num = config.limit_feature_num if "limit_feature_num" in config else 200 # 筛选特征
    sparse_features = df_feature_types[df_feature_types['feature_type'] == 'category']['feature_name'].tolist()[
        :limit_feature_num]
    dense_features = df_feature_types[df_feature_types['feature_type'] == 'dense']['feature_name'].tolist()[
        :limit_feature_num]
    print("num sparse_features is {}".format(len(sparse_features)))
    print("num dense_features is {}".format(len(dense_features)))
    print("num features is ",len(sparse_features)+len(dense_features))
    # 集合特征转化
    lbe_dict = get_lbe_dict(data_merge, sparse_features)
    for feat in lbe_dict:
        lbe = LabelEncoder()
        train_data[feat] = lbe_dict[feat].transform(
            train_data[feat].apply(str))
        dev_data[feat] = lbe_dict[feat].transform(dev_data[feat].apply(str))
        test_data[feat] = lbe_dict[feat].transform(test_data[feat].apply(str))
    if len(dense_features)!=0:
        # dense 特征
        mms = MinMaxScaler(feature_range=(0, 1))
        mms.fit(train_data[dense_features])
        train_data[dense_features] = mms.transform(train_data[dense_features])
        dev_data[dense_features] = mms.transform(dev_data[dense_features])
        test_data[dense_features] = mms.transform(test_data[dense_features])
    else:
        mms = MinMaxScaler(feature_range=(0, 1))
    return sparse_features, dense_features, lbe_dict, mms, train_data, dev_data, test_data