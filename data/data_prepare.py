#%%
import pandas as pd
import numpy as np
import pickle
import argparse
from tqdm import tqdm
import csv
import re
import os
import datetime
from data_utils import construct_AGSR_data,dataframe2dict,build_graph,randomwalkOnWholeGraph,calculate_similarity,construct_BAGSR_data


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',default='demo',help='dataset name:kkbox/jdata2018/demo/')
parser.add_argument('--remove_new_items',action='store_true',help='whether remove new items:if add argument,it will be true') #if add argument,it will be true,no new item
parser.add_argument('--mode',default='BAG_SR',help='which mode:AG_SR/BAG_SR')
opt = parser.parse_args(['--remove_new_items'])


if opt.dataset == 'kkbox':
    dataset = './kkbox/'
    train = pd.read_csv(dataset+'train.csv')
    session_col = 'sessionId'
    item_col = 'song_id'
    operation_col = 'source_screen_name'
elif opt.dataset == 'jdata2018':
    dataset = './jdata2018/'
    session_col = 'sessionId'
    item_col = 'sku_id'
    operation_col = 'type'
    train = pd.read_csv(dataset + 'train.csv', dtype={item_col: str, operation_col: str})
elif opt.dataset == 'demo':
    dataset = './demo/'
    session_col = 'sessionId'
    item_col = 'sku_id'
    operation_col = 'type'
    train = pd.read_csv(dataset + 'train.csv', dtype={item_col: str, operation_col: str})


if opt.remove_new_items:
    test = pd.read_csv(dataset+'test.csv',dtype={item_col:str,operation_col:str})
    dataset += 'no_new_item/'
else:
    test = pd.read_csv(dataset + 'test_new_item.csv',dtype={item_col:str,operation_col:str})
    dataset += 'with_new_item/'

print('--start to load data %s, the time is :%s'%(dataset,str(datetime.datetime.now())))

entity2id = pd.read_csv(dataset+'entity2id',sep = '\t',dtype={'entity':str,'id':int,'type':str})
item2id = entity2id[entity2id['type']=='item']
item2id = dataframe2dict(item2id, 'entity', 'id')
kg2id = pd.read_csv(dataset+'kg2id',sep = '\t',dtype={'head':str,'tail':str,'relation':str})

print('the numbder of entity is :%d'%len(entity2id))
print('the numbder of item is :%d'%len(item2id))
print('the numbder of kg is :%d'%len(kg2id))

if opt.mode == 'AG_SR':
    print('the mode is :',opt.mode)
    data_path = dataset+ 'AG_SR/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    relation2id = pd.read_csv(dataset+'relation2id',sep = '\t',dtype={'head':str,'tail':str,'relation':str})

    train = train[[session_col,item_col]]
    test = test[[session_col,item_col]]
    #uniform cols name
    train.columns = ['sessionId','item']
    test.columns = ['sessionId', 'item']
    print('--%s data have loaded , the time is :%s' % (dataset, str(datetime.datetime.now())))
    train_processed,train_is,test_processed = construct_AGSR_data(train,test,item2id)
    pickle.dump(train_processed,open(data_path+'train_processed.pkl','wb'))
    pickle.dump(test_processed,open(data_path+'test_processed.pkl','wb'))
    """
    print('start to construct item graph',datetime.datetime.now())
    item_graph = build_graph(train_is)
    item_randompath_file = data_path+'item_random_path'
    print('start to random walk on random graph:',datetime.datetime.now())
    randomwalkOnWholeGraph(graph=item_graph, num_walks=30, walk_length=10,randompath_file=item_randompath_file)
    print('randomwalkOnWholeGraph have done')
    item_sim_model_file = data_path+'item_sim.model'
    item_sim_model = calculate_similarity(windowSize=5, dim=160, randompath_file=item_randompath_file, model_file=item_sim_model_file)
    print('item similarity model have done!',datetime.datetime.now())
    """
elif opt.mode == 'BAG_SR':
    print('the mode is :', opt.mode)
    data_path = dataset + 'BAG_SR/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    relation_operation2id = pd.read_csv(dataset+'relation_operation2id',sep='\t',dtype={'head':str,'tail':str,'relation':str})
    operation2id=dataframe2dict(relation_operation2id, 'relation', 'id')
    train = train[[session_col,item_col,operation_col]]
    test = test[[session_col,item_col,operation_col]]
    train.columns = ['sessionId','item','operation']
    test.columns = ['sessionId','item','operation']
    print('--%s data have loaded , the time is :%s' % (dataset, str(datetime.datetime.now())))
    train_processed,train_is,train_os,test_processed = construct_BAGSR_data(train,test,item2id,operation2id)
    pickle.dump(train_processed,open(data_path+'train_processed.pkl','wb'))
    pickle.dump(test_processed,open(data_path+'test_processed.pkl','wb'))












