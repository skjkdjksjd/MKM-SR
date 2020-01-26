import torch
from torch import  nn
import torch.nn.functional as F
import math
from torch.nn import Module,Parameter
from utils_v1 import trans_to_cuda


class GNN(Module):
    def __init__(self,hidden_size,step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size*2
        self.gate_size = 3*hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size,self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size,self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size,self.hidden_size,bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size,self.hidden_size,bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size,self.hidden_size,bias=True)

    def GNN_cell(self,A,hidden):
        input_in = torch.matmul(A[:,:,:A.shape[1]],self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:,:,A.shape[1]:2*A.shape[1]],self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in,input_out],2)
        g_i = F.linear(inputs,self.w_ih,self.b_ih) # batch_size * xx * gate_size
        g_h = F.linear(hidden,self.w_hh,self.b_hh)
        i_r,i_i,i_n = g_i.chunk(3,2) # tensors,chunks,dim
        h_r,h_i,h_n = g_h.chunk(3,2)
        resetgate = torch.sigmoid(i_r+h_r)
        inputgate = torch.sigmoid(i_i+h_i)
        newgate = torch.tanh(i_n + resetgate*h_n)
        hy = newgate + inputgate*(hidden-newgate)
        return hy

    def forward(self,A,hidden):
        for i in range(self.step):
            hidden = self.GNN_cell(A,hidden)
        return hidden


class BAG_SR(Module):
    def __init__(self,opt,n_entity,n_relation,n_item):
        super(BAG_SR, self).__init__()
        self.hidden_size = opt.hidden_size
        self.l2 = opt.l2
        self.n_item = n_item
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.batch_size = opt.batch_size
        self.kg_loss_rate = trans_to_cuda(torch.Tensor([opt.kg_loss_rate]).float())

        self.entity_embedding = nn.Embedding(self.n_entity,self.hidden_size)
        self.relation_embedding = nn.Embedding(self.n_relation,self.hidden_size)
        self.norm_vector = nn.Embedding(self.n_relation,self.hidden_size)

        self.gnn_entity = GNN(self.hidden_size,step=opt.step)
        self.gru_relation = nn.GRU(self.hidden_size,self.hidden_size,num_layers=1,batch_first=True)
        self.linear_one = nn.Linear(self.hidden_size*2,self.hidden_size*2,bias=True)
        self.linear_two = nn.Linear(self.hidden_size*2,self.hidden_size*2,bias=True)
        self.linear_three = nn.Linear(self.hidden_size*2,1,bias=True)
        self.linear_transform = nn.Linear(self.hidden_size*4,self.hidden_size,bias=True)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(),lr = opt.lr,weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=opt.lr_dc_step,gamma = opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size*2)
        for weight in self.parameters():
            ### print('weigght data shape:',weight.data.shape)
            weight.data.uniform_(-stdv, stdv)

    def predict(self, seq_hiddens, masks, itemindexTensor):
        ### print('model predict: masks demo:',masks[:3])
        ### print('masks shape:',masks.shape) #batch_size *L-length
        ### print('torch.sum(mask,1)',torch.sum(masks,1)[:3]) #batch_size * 1
        ht = seq_hiddens[
            torch.arange(masks.shape[0]).long(), torch.sum(masks, 1) - 1]  # the last one #batch_size*hidden_size
        # print('ht demo:',ht[:3],'\h ht shape:',ht.shape)
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size*1*hidden_size
        q2 = self.linear_two(seq_hiddens)  # batch_size*seq_length*hidden_size
        # print('q1.shape:{},q2.shape:{}'.format(q1.shape,q2.shape))
        alpha = self.linear_three(torch.sigmoid(q1 + q2))  # batch_size * seq_len *1
        # print('alpha.shape:',alpha.shape)
        a = torch.sum(alpha * seq_hiddens * masks.view(masks.shape[0], -1, 1).float(), 1)
        # torch.sum((batch_size * seq_len*1)*(batch_size*seq_len*hidden_size)*(batch_size*seq_length*1))
        # a.shape batch_size *hidden_size
        ### print('masks.view(masks.shape[0],-1,1).shape:{}, a.shape:{}'.format((masks.view(masks.shape[0],-1,1)).shape,a.shape))
        a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.entity_embedding.weight[itemindexTensor]  # n_items *latent_size
        scores = torch.matmul(a, b.transpose(1, 0))
        return scores

    def forward(self, inputs, A,relation_inputs):
        entity_hidden = self.entity_embedding(inputs)  # batch,L,hidden_size
        entity_hidden = self.gnn_entity(A, entity_hidden)  # batch,hidden_size?? !todo
      ###  print('entity_hidden.shape:',entity_hidden.shape)
        relation_inputs = self.relation_embedding(relation_inputs)
        relation_output,relation_hidden = self.gru_relation(relation_inputs,None)
       ### print('relation_output.shape:',relation_output.shape)
        return entity_hidden,relation_output

