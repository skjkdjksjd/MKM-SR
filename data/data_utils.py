from tqdm import  tqdm
import pandas as pd
import networkx as nx
import numpy as np
from gensim.models import word2vec
from sklearn.preprocessing import normalize
import pickle


def dataframe2dict(df, key_col, value_col):
    df_dict = {}
    for index,row in df.iterrows():
        key = row[key_col]
        value = row[value_col]
        if key not in df_dict:
            df_dict[key] = value
    return df_dict


def construct_data(data,max_session_length,item2id,item_tail=[0]):
    session_dict = {}
    data_masks = []
    data_padding = []
    data_targets = []
    data_is = []
    for index, row in tqdm(data.iterrows()):
        session_id = row['sessionId']
        item = str(row['item'])
        item = item2id[item] #item2dict key is str
        if session_id not in session_dict:
            session_dict[session_id] = {}
            session_dict[session_id]['items'] = []
        session_dict[session_id]['items'].append(item)
    for sess in tqdm(session_dict.keys()):
        items = session_dict[sess]['items']
        data_is.append(items)
        if len(items) > max_session_length:  # 需要进行截断
            items = items[-max_session_length:]
        for i in range(1, len(items)):
            tar = items[-i]
            inputs = items[:-i]
            mask = [1] * len(inputs) + [0] * (max_session_length - len(inputs))
            data_pad = inputs + item_tail * (max_session_length - len(inputs))
            data_masks.append(mask)
            data_padding.append(data_pad)
            data_targets += [tar]
    data_processed = (data_padding, data_masks, data_targets)
    return data_processed,data_is

def construct_AGSR_data(train,test,item2id,item_tail=[0]):
    session_length = train.groupby('sessionId').size()
    max_session_length = int(session_length.quantile(q=0.99))
    train_processed,train_is = construct_data(train,max_session_length,item2id,item_tail=item_tail)
    test_processed,_ = construct_data(test, max_session_length, item2id, item_tail=item_tail)
    return train_processed,train_is,test_processed

def construct_data_bag(data,max_session_length,item2id,operation2id,item_tail=[0]):
    session_dict = {}
    data_masks = []
    data_paddings = []
    data_operation_paddings = []
    data_targets = []
    data_is = []
    data_os = []
    for index,row in tqdm(data.iterrows()):
        session_id = row['sessionId']
        item = str(row['item'])
        item = item2id[item]
        operation = row['operation']
        operation = operation2id[operation]
        if session_id not in session_dict:
            session_dict[session_id] = {}
            session_dict[session_id]['items'] = []
            session_dict[session_id]['operations'] = []
        session_dict[session_id]['items'].append(item)
        session_dict[session_id]['operations'].append(operation)
    for sess in tqdm(session_dict.keys()):
        items = session_dict[sess]['items']
        operations = session_dict[sess]['operations']
        data_is.append(items)
        data_os.append(operations)
        if len(items) > max_session_length:
            items = items[-max_session_length:]
            operations = operations[-max_session_length:]
        for i in range(1,len(items)):
            tar = items[-i]
            inputs = items[:-i]
            inputs_operations = operations[:-i]
            mask = [1]*len(inputs) + [0]*(max_session_length-len(inputs))
            data_pad = inputs +item_tail *(max_session_length-len(inputs))
            data_op_pad = inputs_operations + item_tail *(max_session_length-len(inputs))
            data_masks.append(mask)
            data_paddings.append(data_pad)
            data_operation_paddings.append(data_op_pad)
            data_targets += [tar]
    data_processed = (data_paddings,data_operation_paddings,data_masks,data_targets)
    return data_processed,data_is,data_os

def construct_BAGSR_data(train,test,item2id,operation2id,item_tail=[0]):
    session_length = train.groupby('sessionId').size()
    max_session_length = int(session_length.quantile(q=0.99))
    train_processed,train_is,train_os = construct_data_bag(train,max_session_length,item2id,operation2id,item_tail)
    test_processed,_,_ = construct_data_bag(test,max_session_length,item2id,operation2id,item_tail)
    return train_processed,train_is,train_os,test_processed






