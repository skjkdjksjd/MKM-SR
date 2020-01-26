import numpy as np
import torch
import datetime
import math
from tqdm import tqdm
from unit_utils import forward_bggnn_model

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

class AG_DATA():
    def __init__(self,data):
        self.data_paddings, self.data_masks, self.data_targets = np.array(data[0]), np.array(data[1]), np.array(data[2])

class BAG_DATA():
    def __init__(self,data):
        self.data_paddings,self.data_operation_paddings, self.data_masks, self.data_targets = np.array(data[0]),np.array(data[1]), np.array(data[2]), np.array(data[3])


def get_slice(slice_index,data_paddings,data_masks,data_targets):
    inputs,masks,targets = data_paddings[slice_index],data_masks[slice_index],data_targets[slice_index]
    items,n_node,A,alias_input = [],[],[],[]
    for u_input in inputs:
        n_node.append(len(np.unique(u_input))) #the length of unique items
    max_n_node = np.max(n_node) #the longest unique item length
    for u_input,u_mask in zip(inputs,masks):
        node = np.unique(u_input) #the unique items of inputs
        items.append(node.tolist()+(max_n_node-len(node))*[0]) #items list
        u_A = np.zeros((max_n_node,max_n_node))
        for i in range(len(u_input)-1):
            if u_input[i+1] == 0:
                break
            u = np.where(node == u_input[i])[0][0] #np.where return a tuple,so need use [0][0] to show the value
            v = np.where(node == u_input[i+1])[0][0]
            u_A[u][v] +=1 # different from sr-gnn
        ### print('u_A raw:',u_A)
        u_sum_in = np.sum(u_A,0) # in degree
        u_sum_in[np.where(u_sum_in == 0)] = 1
        ### print('u_sum_in',u_sum_in)
        u_A_in = np.divide(u_A,u_sum_in)
        ### print('u_A_in:\n',u_A_in)

        u_sum_out = np.sum(u_A,1) #out degree
        u_sum_out[np.where(u_sum_out ==0)] = 1
        ### print('u_sum_out:',u_sum_out)
        u_A_out = np.divide(u_A.T,u_sum_out)
        ### print('u_A_out:\n', u_A_out)
        u_A = np.concatenate([u_A_in,u_A_out]).T
        A.append(u_A) # can't be array,is irregular
        alias_input.append([np.where(node == i)[0][0] for i in u_input] )
    return alias_input,A,items,masks,targets

def get_bag_slice(slice_index,data_paddings,data_operation_paddings,data_masks,data_targets):
    alias_input, A, items, masks, targets = get_slice(slice_index,data_paddings,data_masks,data_targets)
    operation_inputs = data_operation_paddings[slice_index]
    return alias_input,A,items,operation_inputs,masks,targets

def generate_batch_slices(len_data,shuffle=True,batch_size=128): #padding,masks,targets
    n_batch = math.ceil(len_data / batch_size)
    shuffle_args = np.arange(n_batch*batch_size)
    if shuffle:
        np.random.shuffle(shuffle_args)
    slices = np.split(shuffle_args,n_batch)
    slices = [i[i<len_data] for i in slices]
    return slices

def forward_model(model,slice_index,data,itemindexTensor):
    alias_inputs,A,items,masks,targets = get_slice(slice_index,data.data_paddings,data.data_masks,data.data_targets)
    ### print('alias inputs demo',alias_inputs[:3])
    ###print('len alias inputs',len(alias_inputs))
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    masks = trans_to_cuda(torch.Tensor(masks).long())
    hidden = model.forward(items,A) # batch_size * max_n_node * hidden_size
   # print('hidden shape:',hidden.shape)
    get = lambda i:hidden[i][alias_inputs[i]]
    seq_hiddens = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()]) #batch_size*L=length*hidden_size # todo
    #print('seq_hiddens shape',seq_hiddens.shape)
    return targets,model.predict(seq_hiddens,masks,itemindexTensor),masks
"""
def get_pack_padded_sequence(inputs,masks,input_embedding):
   # print('get pack_padded:')
   # print('inputs:', inputs)
    lens = torch.sum(masks,dim=1).flatten()
   # print('lens2:',lens)
    embed_input = input_embedding(inputs)
    inputs_ = torch.nn.utils.rnn.pack_padded_sequence(embed_input,lens,enforce_sorted=False,batch_first=True)
    return inputs_
"""
def forward_bag_model(model,slice_index,data,itemindexTensor):
    #print('slice length is:',len(slice_index),datetime.datetime.now())
    alias_inputs,A,items,operation_inputs,masks,targets = get_bag_slice(slice_index, data.data_paddings,data.data_operation_paddings, data.data_masks, data.data_targets)
    #print('data have get slice,', datetime.datetime.now())
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    #print('operation_inputs:',operation_inputs)
    operation_inputs = trans_to_cuda(torch.Tensor(operation_inputs).long())
    masks = trans_to_cuda(torch.Tensor(masks).long())
    #print('data have trans to cuda,', datetime.datetime.now())
    #inputs_ =  get_pack_padded_sequence(operation_inputs,masks,model.relation_embedding)
    entity_hidden,relation_hidden = model.forward(items, A,operation_inputs)
    #print('model have forward,', datetime.datetime.now())
    get = lambda i: entity_hidden[i][alias_inputs[i]]
    seq_hiddens = torch.stack(
        [get(i) for i in torch.arange(len(alias_inputs)).long()])  # batch_size*L-length*hidden_size # todo
    #### print('seq_hiddens shape',seq_hiddens.shape)
    seq_hiddens = torch.cat([seq_hiddens,relation_hidden],dim=2)
    #### print('seq_hiddens shape new', seq_hiddens.shape)
    #print('seq hidden have done,', datetime.datetime.now())
    state = model.predict(seq_hiddens, masks, itemindexTensor)
    #print('model have predict;',datetime.datetime.now())
    return targets, state, masks

def train_predict(model,train_data,test_data,item_ids,itemid2index,kg2id):
    model.train()
    hs = kg2id['head'].values
    ts = kg2id['tail'].values
    rs = kg2id['relation'].values
    kg_slices = generate_batch_slices(len(hs),shuffle=True,batch_size=model.batch_size*2)
    kg_total_loss = 0
    for slice in tqdm(kg_slices):
        model.optimizer.zero_grad()
        batch_h = trans_to_cuda(torch.Tensor(hs[slice]).long())
        batch_t = trans_to_cuda(torch.Tensor(ts[slice]).long())
        batch_r = trans_to_cuda(torch.Tensor(rs[slice]).long())
        kg_loss = model.get_kg_loss(batch_h,batch_t,batch_r)+model.l2*model.kg_regularization(batch_h,batch_t,batch_r)
        kg_loss.backward()
        model.optimizer.step()
        kg_total_loss += kg_loss.item()
    print('now the kg total loss is:%.4f'%(kg_total_loss))

    itemindexTensor = trans_to_cuda(torch.Tensor(item_ids).long())

    print('start training: ', datetime.datetime.now())
    total_loss = 0.0
    slices = generate_batch_slices(len(train_data.data_paddings), shuffle=True, batch_size=model.batch_size)
    index = 0
    for slice_index,j in zip(slices,np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets,scores,masks = forward_model(model,slice_index,train_data,itemindexTensor)
        targets = [itemid2index[tar] for tar in targets]
        targets = trans_to_cuda(torch.Tensor(targets).long())
        # print('forward model,len score is %d,len score 0 is %d,len batch size is%d' % (
        # len(scores), len(scores[0]), len(slice_index)), datetime.datetime.now())
        # print("scores 0:", scores[0], '\n targets 0:', targets[0])
        # print("scores 1:", scores[1], '\n targets 0:', targets[1])
        loss =trans_to_cuda(model.loss_function(scores,targets))

        #print("the slice index is %d,the loss is %.4f" % (index, loss))
        index += 1
       # print('test1:', model.entity_embedding.weight[1:5])
        loss.backward()
        #print('test2:', model.entity_embedding.weight[1:5])
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()),datetime.datetime.now())
    #model.scheduler.step()
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = generate_batch_slices(len(test_data.data_paddings), shuffle=False, batch_size=model.batch_size)
    for slice_index in slices:
        targets, scores,masks = forward_model(model, slice_index, test_data,itemindexTensor)
        sub_scores = scores.topk(20)[1]  #tensor has the top_k functions
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        targets = [itemid2index[tar] for tar in targets]
        #print('sub scores:',sub_scores,'targets:',targets)
        for score,target,mask in zip(sub_scores,targets,masks):
            hit.append(np.isin(target,score))
            if len(np.where(score == target)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1/(np.where(score==target)[0][0]+1))

    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr


def train_predict_bag(model,train_data,test_data,item_ids,itemid2index):
    print('enter the model,',datetime.datetime.now())
    itemindexTensor = torch.Tensor(item_ids).long()
    print('item id to tensor,',datetime.datetime.now())
    print('start training: ', datetime.datetime.now())
    total_loss = 0.0
    slices = generate_batch_slices(len(train_data.data_paddings), shuffle=True, batch_size=model.batch_size)
    index = 0
    model.train()
    for slice_index, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        #print('the slice start time:',datetime.datetime.now())
        targets, scores, masks = forward_bag_model(model, slice_index, train_data, itemindexTensor)
        #print('the slice end time:',datetime.datetime.now())
        targets = [itemid2index[tar] for tar in targets]
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets)
        index += 1
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % 100 == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()),datetime.datetime.now())
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = generate_batch_slices(len(test_data.data_paddings), shuffle=False, batch_size=model.batch_size)
    for slice_index in slices:
        targets, scores, masks = forward_bag_model(model, slice_index, test_data, itemindexTensor)
        sub_scores = scores.topk(20)[1]  # tensor has the top_k functions
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        targets = [itemid2index[tar] for tar in targets]
        for score, target, mask in zip(sub_scores, targets, masks):
            hit.append(np.isin(target, score))
            if len(np.where(score == target)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target)[0][0] + 1))

    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr


def train_predict_mat(model,train_data,test_data,item_ids,itemid2index,kg2id):
    model.train()
    hs = kg2id['head'].values
    ts = kg2id['tail'].values
    rs = kg2id['relation'].values
    kg_slices = generate_batch_slices(len(hs), shuffle=True, batch_size=model.batch_size * 2)
    kg_total_loss = 0
    for slice in tqdm(kg_slices):
        model.optimizer.zero_grad()
        batch_h = trans_to_cuda(torch.Tensor(hs[slice]).long())
        batch_t = trans_to_cuda(torch.Tensor(ts[slice]).long())
        batch_r = trans_to_cuda(torch.Tensor(rs[slice]).long())
        kg_loss = model.get_kg_loss(batch_h, batch_t, batch_r) + model.l2 * model.kg_regularization(batch_h, batch_t,
                                                                                                    batch_r)
        kg_loss.backward()
        model.optimizer.step()
        kg_total_loss += kg_loss.item()
    print('now the kg total loss is:%.4f' % (kg_total_loss))

    itemindexTensor = trans_to_cuda(torch.Tensor(item_ids).long())
    print('start training: ', datetime.datetime.now())
    total_loss = 0.0
    slices = generate_batch_slices(len(train_data.data_paddings), shuffle=True, batch_size=model.batch_size)
    index = 0
    for slice_index, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores, masks = forward_bggnn_model(model, slice_index, train_data, itemindexTensor)
        targets = [itemid2index[tar] for tar in targets]
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = trans_to_cuda(model.loss_function(scores, targets))
        index += 1
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % 100 == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()), datetime.datetime.now())
    print('\tLoss:\t%.3f' % total_loss)
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = generate_batch_slices(len(test_data.data_paddings), shuffle=False, batch_size=model.batch_size)
    for slice_index in slices:
        targets, scores, masks = forward_bggnn_model(model, slice_index, test_data, itemindexTensor)
        sub_scores = scores.topk(20)[1]  # tensor has the top_k functions
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        targets = [itemid2index[tar] for tar in targets]
        for score, target, mask in zip(sub_scores, targets, masks):
            hit.append(np.isin(target, score))
            if len(np.where(score == target)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target)[0][0] + 1))

    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr

def get_slice_cotrain(slice_index,data_paddings,data_masks,data_targets):
    inputs,masks,targets = data_paddings[slice_index],data_masks[slice_index],data_targets[slice_index]
    slice_items = inputs.flatten()[inputs.flatten()!=0]
    items,n_node,A,alias_input = [],[],[],[]
    for u_input in inputs:
        n_node.append(len(np.unique(u_input))) #the length of unique items
    max_n_node = np.max(n_node) #the longest unique item length
    for u_input,u_mask in zip(inputs,masks):
        node = np.unique(u_input) #the unique items of inputs
        items.append(node.tolist()+(max_n_node-len(node))*[0]) #items list
        u_A = np.zeros((max_n_node,max_n_node))
        for i in range(len(u_input)-1):
            if u_input[i+1] == 0:
                break
            u = np.where(node == u_input[i])[0][0] #np.where return a tuple,so need use [0][0] to show the value
            v = np.where(node == u_input[i+1])[0][0]
            u_A[u][v] +=1 # different from sr-gnn
        ### print('u_A raw:',u_A)
        u_sum_in = np.sum(u_A,0) # in degree
        u_sum_in[np.where(u_sum_in == 0)] = 1
        ### print('u_sum_in',u_sum_in)
        u_A_in = np.divide(u_A,u_sum_in)
        ### print('u_A_in:\n',u_A_in)

        u_sum_out = np.sum(u_A,1) #out degree
        u_sum_out[np.where(u_sum_out ==0)] = 1
        ### print('u_sum_out:',u_sum_out)
        u_A_out = np.divide(u_A.T,u_sum_out)
        ### print('u_A_out:\n', u_A_out)
        u_A = np.concatenate([u_A_in,u_A_out]).T
        A.append(u_A) # can't be array,is irregular
        alias_input.append([np.where(node == i)[0][0] for i in u_input] )
    return alias_input,A,items,masks,targets,slice_items

def forward_model_cotrain(model,slice_index,data,itemindexTensor,kg2id):
    alias_inputs,A,items,masks,targets,slice_items = get_slice_cotrain(slice_index,data.data_paddings,data.data_masks,data.data_targets)
    kg_slice = kg2id[np.in1d(kg2id['head'].values,slice_items)]
    hs = kg_slice['head'].values
    ts = kg_slice['tail'].values
    rs = kg_slice['relation'].values
    batch_h = trans_to_cuda(torch.Tensor(hs).long())
    batch_t = trans_to_cuda(torch.Tensor(ts).long())
    batch_r = trans_to_cuda(torch.Tensor(rs).long())
    kg_loss = model.kg_loss_rate*(model.get_kg_loss(batch_h, batch_t, batch_r) +
                                 model.l2 * model.kg_regularization(batch_h, batch_t, batch_r))
    ### print('alias inputs demo',alias_inputs[:3])
    ###print('len alias inputs：',len(alias_inputs))
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    masks = trans_to_cuda(torch.Tensor(masks).long())
    hidden = model.forward(items,A) # batch_size * max_n_node * hidden_size
   # print('hidden shape:',hidden.shape)
    get = lambda i:hidden[i][alias_inputs[i]]
    seq_hiddens = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()]) #batch_size*L=length*hidden_size # todo
    #print('seq_hiddens shape',seq_hiddens.shape)
    return targets,model.predict(seq_hiddens,masks,itemindexTensor),masks,kg_loss



def train_predict_cotrain(model,train_data,test_data,item_ids,itemid2index,kg2id):
    print('start training: ', datetime.datetime.now())
    itemindexTensor = torch.Tensor(item_ids).long()
    total_loss = 0.0
    slices = generate_batch_slices(len(train_data.data_paddings), shuffle=True, batch_size=model.batch_size)
    index = 0
    for slice_index,j in zip(slices,np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets,scores,masks,kg_loss = forward_model_cotrain(model,slice_index,train_data,itemindexTensor,kg2id)
        targets = [itemid2index[tar] for tar in targets]
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores,targets)+kg_loss
        index += 1
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    #model.scheduler.step()
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr = [], []
    slices = generate_batch_slices(len(test_data.data_paddings), shuffle=False, batch_size=model.batch_size)
    for slice_index in slices:
        targets, scores,masks,_ = forward_model_cotrain(model, slice_index, test_data,itemindexTensor,kg2id)
        sub_scores = scores.topk(20)[1]  #tensor has the top_k functions
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        targets = [itemid2index[tar] for tar in targets]
        #print('sub scores:',sub_scores,'targets:',targets)
        for score,target,mask in zip(sub_scores,targets,masks):
            hit.append(np.isin(target,score))
            if len(np.where(score == target)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1/(np.where(score==target)[0][0]+1))

    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr