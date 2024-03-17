import sys; sys.path.append('../noise')
import noise_matrix
from noise_matrix import sig_t
import argparse
from models import *
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import numpy as np
import wandb
# pytorch libraries
import torch
from torch import optim,nn
import time
import random
from data import *
from configs import get_args
import torch.nn.functional as F
from models import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_loss_list = []
val_acc_list = []
test_acc_list = []

def train_model(model,
                trans,
                train_data_loader,
                valid_data_loader,
                test_data_loader,
                optimizer_es,
                optimizer_trans,
                scheduler1,
                scheduler2,
                n_epochs,
                loss_func_ce,
                batch_size,
                lr,
                epsilon,
                model_name,
                train_transition_matrix):


    for epoch in range(args.n_epochs):

        print('epoch {}'.format(epoch + 1))

        epoch_time_start = time.time()

        model.train()
        trans.train()

        train_loss = 0.
        train_vol_loss =0.
        train_acc = 0.
        val_loss = 0.
        val_acc = 0.
        eval_loss = 0.
        eval_acc = 0.

        train_items = 0
        for batch_x, batch_y in train_data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            train_items += len(batch_x)
            optimizer_es.zero_grad()
            optimizer_trans.zero_grad()


            clean = F.softmax(model(batch_x), 1)
            t = trans()

            out = torch.mm(clean, t)

            vol_loss = t.slogdet().logabsdet

            ce_loss = loss_func_ce(out.log(), batch_y.long())
            loss = ce_loss + lam * vol_loss

            train_loss += loss.item()
            train_vol_loss += vol_loss.item()

            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()


            loss.backward()
            optimizer_es.step()
            optimizer_trans.step()
        print('Train Loss: {:.6f}, Vol_loss: {:.6f}  Acc: {:.6f}'.format(train_loss / train_items, train_vol_loss / train_items, train_acc / train_items))

        # --------- Estimation error ---------- #
        est_T = t.detach().cpu().numpy()
        estimate_error = noise_matrix.error(est_T, train_transition_matrix)

        matrix_path = f'./ckpts/trans_matrix_est/eps_{epsilon}/{model_name}_lam_{lam}_epochs_{n_epochs}_lr_{lr}_bs_{batch_size}_seed_{args.seed}_matrix_epoch_{epoch+1}.npy'
        np.save(matrix_path, est_T)

        print('Estimation Error: {:.2f}'.format(estimate_error))
        print(est_T)

        # ----------------------------------- #


        scheduler1.step()
        scheduler2.step()

        valid_items = 0
        with torch.no_grad():
            model.eval()
            trans.eval()
            for batch_x, batch_y in valid_data_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                valid_items += len(batch_x)
                clean =  F.softmax(model(batch_x))
                t = trans()

                out = torch.mm(clean, t)
                loss = loss_func_ce(out.log(), batch_y.long())
                val_loss += loss.item()
                pred = torch.max(out, 1)[1]
                val_correct = (pred == batch_y).sum()
                val_acc += val_correct.item()

                
        print('Val Loss: {:.6f}, Acc: {:.6f}'.format(val_loss / valid_items, val_acc / valid_items))

        test_items = 0
        with torch.no_grad():
            model.eval()
            trans.eval()

            for batch_x, batch_y in test_data_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                test_items += len(batch_x)
                clean = F.softmax(model(batch_x))

                loss = loss_func_ce(clean.log(), batch_y.long())
                eval_loss += loss.item()
                pred = torch.max(clean, 1)[1]
                eval_correct = (pred == batch_y).sum()
                eval_acc += eval_correct.item()

            print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / test_items,
                                                          eval_acc / test_items))


        epoch_time_end = time.time()
        # Log 
        wandb.log({'epoch': epoch,
                   'epoch_time': epoch_time_end-epoch_time_start,
                   'test_estimation_error': estimate_error,
                   'test_acc': eval_acc / test_items,
                   'test_loss': eval_loss / test_items,
                   'valid_acc': val_acc / valid_items,
                   'valid_loss': val_loss / valid_items,
                   'train_acc': train_acc / train_items,
                   'train_loss': train_loss / train_items})

        if epoch <= 9:
            best_ckpt_epoch = epoch + 1
            best_model_path = f'./ckpts/trans_matrix_est/eps_{epsilon}/new_model_{model_name}_{n_epochs}_lr_{lr}_bs_{batch_size}_lam_{lam}.pth'
            torch.save(model, best_model_path)
        
        val_loss_list.append(val_loss / valid_items)
        val_acc_list.append(val_acc / valid_items)
        test_acc_list.append(eval_acc / test_items)



    val_loss_array = np.array(val_loss_list)
    val_acc_array = np.array(val_acc_list)
    model_index = np.argmin(val_loss_array)
    model_index_acc = np.argmax(val_acc_array)

    matrix_path = f'./ckpts/trans_matrix_est/eps_{epsilon}/{model_name}_lam_{lam}_epochs_{n_epochs}_lr_{lr}_bs_{batch_size}_seed_{args.seed}_' 'matrix_epoch_%d.npy' % (model_index+1)
    final_est_T = np.load(matrix_path)
    final_estimate_error = noise_matrix.error(final_est_T, train_transition_matrix)

    matrix_path_acc = f'./ckpts/trans_matrix_est/eps_{epsilon}/{model_name}_lam_{lam}_epochs_{n_epochs}_lr_{lr}_bs_{batch_size}_seed_{args.seed}_' + 'matrix_epoch_%d.npy' % (model_index_acc+1)
    final_est_T_acc = np.load(matrix_path_acc)
    final_estimate_error_acc = noise_matrix.error(final_est_T_acc, train_transition_matrix)

    print("Final test accuracy: %f" % test_acc_list[model_index])
    print("Final test accuracy acc: %f" % test_acc_list[model_index_acc])
    print("Final estimation error loss: %f" % final_estimate_error)
    print("Final estimation error loss acc: %f" % final_estimate_error_acc)
    print("Best epoch: %d" % model_index)
    print(final_est_T)


def main(args):
    print (f'---> Noise level {args.epsilon}')

    train_data_loader, \
    valid_data_loader, \
    test_data_loader, \
    (train_transition_matrix, valid_transition_matrix) =  get_cxr14_data_transition_matrix(batch_size=args.batch_size, 
                                                                                         train_epsilon=args.epsilon,
                                                                                         valid_epsilon=args.epsilon,
                                                                                         noise_type=noise_type)
    model = get_model(args.model_name)
    model.to(device)

    trans = sig_t(device, 15)

    trans = trans.to(device)

    #optimizer and StepLR
    milestones = [30,60]
    optimizer_trans = optim.SGD(trans.parameters(), lr=iam_lr, weight_decay=0, momentum=0.9)
    # optimizer_trans = optim.Adam(trans.parameters(), lr=iam_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    scheduler2 = MultiStepLR(optimizer_trans, milestones=[50, 80], gamma=0.1)
    
    
    loss_func_ce = torch.nn.NLLLoss()
    for lr in [0.00001]:
        args.lr = lr
        if args.opt == 'Adam':
            optimizer_es = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
        elif args.opt == 'SGD':
            optimizer_es = optim.SGD(model.parameters(), lr=args.lr, weight_decay=weight_decay, momentum=0.9)
        scheduler1 = MultiStepLR(optimizer_es, milestones=milestones, gamma=0.1)


        with wandb.init(project='cxr14_with_noisy_labels_using_T_estimate'):
            wandb.config.update(args)
            print ('=== Start Training ===')
            best_model_path = train_model(model = model, 
                                    trans = trans,
                                    train_data_loader = train_data_loader, 
                                    valid_data_loader = valid_data_loader,
                                    test_data_loader = test_data_loader, 
                                    optimizer_es = optimizer_es,
                                    optimizer_trans = optimizer_trans,
                                    scheduler1 = scheduler1,
                                    scheduler2 = scheduler2,
                                    n_epochs = args.n_epochs, 
                                    loss_func_ce = loss_func_ce, 
                                    batch_size = args.batch_size,
                                    lr=args.lr,
                                    epsilon=args.epsilon,
                                    model_name= args.model_name, 
                                    train_transition_matrix = train_transition_matrix)

            print ('=== End Training ===')



if __name__ == "__main__":
    args = get_args()

    noise_type = 'symmetric' # 'asymmetric'
    weight_decay = 1e-4
    lam = args.lam
    iam_lr = args.lr_lam

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    main(args)
