import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy
import pickle
from model import *
import argparse
from utils import load_citation, accuracy
import torch.optim as optim
import nni
import time
import numpy as np
import statistics
from prettytable import PrettyTable

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--seed', type=int, default=2020, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--layer', type=int, default=3, help='Number of layers.')
# parser.add_argument('--hidden', type=int, default= 512, help='hidden dimensions.')
parser.add_argument('--patience', type=int, default=100, help='Patience')  # 100
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--variant', action='store_true', default=False, help='GCN* model.')
parser.add_argument('--test', action='store_true', default=False, help='evaluation on test set.')
args = parser.parse_args()
print(args)

path = '/home/ourbelovedsejutimada/Documents/Swakshar/GA-GWNN/'

support00 = np.load(path+'supports/'+args.data+'_support00.npy')
support01 = np.load(path+'supports/'+args.data+'_support01.npy')
support10 = np.load(path+'supports/'+args.data+'_support10.npy')
support11 = np.load(path+'supports/'+args.data+'_support11.npy')

# support00 = np.transpose(support00)
# support10 = np.transpose(support10)
# np.save('cora_support00_raf.npy',support00)
# np.save('cora_support10_raf.npy',support10)


adj, features, labels, idx_train, idx_val, idx_test, max_degree = load_citation(args.data)
cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)
features = features.to(device)
adj = adj.to(device)
support0 = torch.sparse_coo_tensor(support00, support01, (adj.shape[0], adj.shape[0]))
support1 = torch.sparse_coo_tensor(support10, support11, (adj.shape[0], adj.shape[0]))
support0 = support0.to(device).to(torch.float32)
support1 = support1.to(device).to_dense().to(torch.float32)

time_counter = []
def build_and_train(hype_space):
    modelname = 'nof'
    modeltype = {'nof':nof,'GWCNII': GWCNII, 'GCNII': GCNII, 'combine': combinemodel, }
    #np.random.seed(args.seed)
    #torch.manual_seed(args.seed)
    model = modeltype[modelname](
        mydev=device,
        myf=hype_space['myf'],
        max_degree=max_degree,
        support0=support0,
        support1=support1,
        adj=adj,
        gamma=hype_space['gamma'],
        nnode=features.shape[0],
        nfeat=features.shape[1],
        nlayers=args.layer,
        nhidden=hype_space['hidden'],
        nclass=int(labels.max()) + 1,
        dropout=hype_space['dropout'],
        lamda=hype_space['lambda'],
        alpha=hype_space['alpha'],
        variant=args.variant).to(device)
    
    optimizer = optim.Adam([{'params': model.params1, 'weight_decay': hype_space['wd1']},
                            {'params': model.params2, 'weight_decay': hype_space['wd2']},],
                           lr=0.1*hype_space['lr_rate_mul'])
    
    # def count_parameters(model):
    #     table = PrettyTable(["Modules", "Parameters"])
    #     total_params = 0
    #     for name, parameter in model.named_parameters():
    #         if not parameter.requires_grad: continue
    #         params = parameter.numel()
    #         table.add_row([name, params])
    #         total_params+=params
    #     print(table)
    #     print(f"Total Trainable Params: {total_params}")
    #     return total_params

    # count_parameters(model)
    
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
            best_val_acc = acc_val
            #if acc_tes > best_test_acc:
            best_test_acc = acc_tes
            best_test_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1
            
        #if acc_tes > best_test_acc:
            #best_test_acc = acc_tes
            #best_test_epoch = epoch

        print('Epoch:%4d|train loss:%.3f acc:%.2f|val loss:%.3f acc:%.2f|test loss:%.3f acc:%.2f|best acc:%.2f epoch:%d'
                %(epoch + 1,loss_tra,acc_tra * 100,loss_val,acc_val * 100,loss_tes,acc_tes*100,best_test_acc*100,best_test_epoch))
        print('{} seconds'.format(time.time() - t0))
        time_counter.append(time.time()-t0)
        if bad_counter == args.patience:
            print('early stopping at epoch %d'%epoch)
            break
    nni.report_final_result(best_test_acc)
    return best_test_acc

t_total = time.time()
SEEDS=[1941488137,4198936537,983997831,4023022230,4019585660,2108550661,1648766618,625014529,3212139042,2424918363]
if __name__ == "__main__":
    acc_list = []
    for i in range(10):
        t1=time.time()
        print("epoch = ", i)
        np.random.seed(SEEDS[i])
        torch.manual_seed(SEEDS[i])
        params={"lr_rate_mul":0.3,"alpha":0.1,"lambda":0.5,"gamma":0.2,"wd1":0.05,"wd2":0.0009,"dropout":0.7,"hidden":64,"myf":0.5}
        #params={"lr_rate_mul":0.3,"alpha":0.1,"lambda":0.5,"gamma":0.2,"wd1":0.05,"wd2":0.0009,"dropout":0.7,"hidden":64,"myf":0.5}
        acc = build_and_train(params)
        t2=time.time()
        acc_list.append(acc)
        print('time= %.2f s'%(t2-t1))
        # params=nni.get_next_parameter()
        # build_and_train(params)
        print("average time per epoach: ", statistics.mean(time_counter))
    print("Test acc.:{:.4f}".format(np.mean(acc_list)))
    print("Test std.:{:.4f}".format(np.std(acc_list)))