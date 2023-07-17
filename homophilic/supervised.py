from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from process import *
from utils import *
from model import *
import uuid
import nni
from sklearn.preprocessing import normalize
import statistics

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--seed', type=int, default=2020, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help= 'Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--layer', type=int, default=2, help='Number of layers.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--variant', action='store_true', default=False, help=
    'GCN* model.')
parser.add_argument('--test', action='store_true', default=False, help=
    'evaluation on test set.')
args = parser.parse_args()
print(args)

#support00 = np.transpose(support00)
#np.save('chameleon_support00.npy',support00)
#support10 = np.transpose(support10)
#np.save('chameleon_support10.npy',support10)

cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)
checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'
print(cudaid,checkpt_file)

#datastr = args.data
#splitstr = 'splits/'+args.data+'_split_0.6_0.2_0'+'.npz'
#adj_, _, _, _, _, _, _, _ = full_load_data(datastr,splitstr)

path = '/home/ourbelovedsejutimada/Documents/Swakshar/GA-GWNN/'
def build_and_train(hype_space,datastr, splitstr,i):
    adj, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = full_load_data(datastr,splitstr)
    
    adj = adj.to(device)
    features = features.to(device)
    modelname = 'nof'
    #modelname = 'GWCNII'
    modeltype = {'nof':nof,'GWCNII': GWCNII, 'GCNII': GCNII, 'combine': combinemodel, }
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = modeltype[modelname](
        mydev=device,
        adj=adj,
        nnode=features.shape[0],
        nfeat=features.shape[1],
        nlayers=args.layer,
        nhidden=hype_space['hidden'],
        nclass=int(labels.max()) + 1,
        dropout=hype_space['dropout'],
        variant=args.variant).to(device)
    
    # model = modeltype[modelname](
    #     mydev=device,
    #     myf=hype_space['myf'],
    #     support0=support0,
    #     support1=support1,
    #     nnode=features.shape[0],
    #     nfeat=features.shape[1],
    #     nlayers=args.layer,
    #     nhidden=hype_space['hidden'],
    #     nclass=int(labels.max()) + 1,
    #     dropout=hype_space['dropout'],
    #     lamda=hype_space['lambda'],
    #     alpha=hype_space['alpha'],
    #     variant=args.variant).to(device)
    
        
    optimizer = optim.Adam([{'params': model.params1, 'weight_decay': hype_space['wd1']},
                            {'params': model.params2, 'weight_decay': hype_space['wd2']},],
                           lr=0.1*hype_space['lr_rate_mul'])

    def train():
        model.train()
        optimizer.zero_grad()
        output=model(features)
        acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
        loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
        loss_train.backward()
        optimizer.step()
        return loss_train.item(), acc_train.item()

    def validate():
        model.eval()
        with torch.no_grad():
            output=model(features)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
            acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
            return loss_val.item(), acc_val.item()

    def test():
        # model.load_state_dict(torch.load(checkpt_file))
        model.eval()
        with torch.no_grad():
            output=model(features)
            loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
            acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
            return loss_test.item(), acc_test.item()

    #adj, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = full_load_data(datastr,splitstr)
    bad_counter = 0
    best = 999999999
    best_test_acc = 0
    best_val_acc = 0
    lrnow = optimizer.state_dict()['param_groups'][0]['lr']
    for epoch in range(args.epochs):
        t0 = time.time()
        loss_tra, acc_tra = train()
        loss_val,acc_val = validate()
        loss_tes,acc_tes = test()
        # nni.report_intermediate_result(acc_tes)

        if epoch==0 or loss_val < best:
            best = loss_val
            best_epoch = epoch
            acc = acc_val
            #if acc_tes > best_test_acc:
            best_test_acc = acc_tes
            best_test_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        # if acc_val > best_val_acc:
        #     best_test_acc = acc_tes
        #     best_val_acc = acc_val
        #     best_test_epoch = epoch
        #     bad_counter = 0
        # else:
        #     bad_counter += 1

        print('Epoch:%4d|train loss:%.3f acc:%.2f|val loss:%.3f acc:%.2f|test loss:%.3f acc:%.2f|best acc:%.4f epoch:%d'
                %(epoch + 1,loss_tra,acc_tra * 100,loss_val,acc_val * 100,loss_tes,acc_tes*100,best_test_acc*100,best_test_epoch))
        print('{} seconds'.format(time.time() - t0))
        if bad_counter == args.patience:
            print('early stopping at epoch %d'%epoch)
            break
    nni.report_final_result(best_test_acc)
    return best_test_acc

t_total = time.time()
#avg_acc_list = []
#for j in range(10):
acc_list = []
for i in range(10):
    datastr = args.data
    splitstr = path+'splits/'+args.data+'_split_0.6_0.2_'+str(i)+'.npz'
    #acc_list.append(trainer(datastr,splitstr))
    #params={"lr_rate_mul":1,"alpha":0.1,"lambda":1,"gamma":0.2,"wd1":0.05,"wd2":0.0005,"dropout":0.5,"hidden":64,"myf":1.2}
    params={"lr_rate_mul":0.1, "wd1":0.05, "wd2":0.00019, "dropout":0.55, "hidden":64}
    acc = build_and_train(params, datastr, splitstr,str(i))
    acc_list.append(acc)
    print(i,": {:.2f}".format(acc_list[-1]))
#print("Train cost: {:.4f}s".format(time.time() - t_total))
print("Test acc.:{:.4f}".format(np.mean(acc_list)))

print("Test std.:{:.4f}".format(np.std(acc_list)))
    
#     avg_acc = np.mean(acc_list)
#     avg_acc_list.append(avg_acc)
    
# print("Test acc.:{:.4f}".format(np.mean(avg_acc_list)))

# print("Test std.:{:.4f}".format(np.std(avg_acc_list)))
#statistics.stdev(A_rank)