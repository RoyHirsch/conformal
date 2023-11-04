import os
import pickle
import numpy as np
from scipy.special import softmax

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def save_pickle(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


def get_logits_dataloader(logits, labels, batch_size=64, shuffle=False, pin_memory=True):
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(logits),
                                            torch.from_numpy(labels).long()) 
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             pin_memory=pin_memory)
    return dataloader


def platt_logits(calib_loader, max_iters=100, lr=0.01, epsilon=0.005):
    nll_criterion = nn.CrossEntropyLoss().cuda()

    T = nn.Parameter(torch.Tensor([1.]).cuda())
    optimizer = optim.SGD([T], lr=lr)
    for iter in tqdm(range(max_iters)):
        T_old = T.item()
        for x, targets in calib_loader:
            optimizer.zero_grad()
            x = x.cuda()
            x.requires_grad = True
            out = x/T
            loss = nll_criterion(out, targets.long().cuda())
            loss.backward()
            optimizer.step()
        if abs(T_old - T.item()) < epsilon:
            break
    return T.item() 


def conformal_calibration_logits(calib_loader, T=1., alpha=.1, randomized=True, allow_zero_sets=True):
    with torch.no_grad():
        E = np.array([])
        for logits, targets in calib_loader:
            logits = logits.detach().cpu().numpy()

            scores = softmax(logits/T, axis=1)

            I, ordered, cumsum = sort_sum(scores)

            E = np.concatenate((E,giq(scores,targets,I=I,ordered=ordered,cumsum=cumsum ,randomized=randomized, allow_zero_sets=allow_zero_sets)))
            
        Qhat = np.quantile(E,1-alpha,interpolation='higher')

        return Qhat 


def sort_sum(scores):
    I = scores.argsort(axis=1)[:,::-1]
    ordered = np.sort(scores,axis=1)[:,::-1]
    cumsum = np.cumsum(ordered,axis=1) 
    return I, ordered, cumsum


def giq(scores, targets, I, ordered, cumsum, randomized, allow_zero_sets):
    """
        Generalized inverse quantile conformity score function.
        E from equation (7) in Romano, Sesia, Candes.  Find the minimum tau in [0, 1] such that the correct label enters.
    """
    E = -np.ones((scores.shape[0],))
    for i in range(scores.shape[0]):
        E[i] = get_tau(scores[i:i+1,:],targets[i].item(),I[i:i+1,:],ordered[i:i+1,:],cumsum[i:i+1,:],randomized=randomized, allow_zero_sets=allow_zero_sets)

    return E


def gcq(scores, tau, I, ordered, cumsum, randomized, allow_zero_sets):
    sizes_base = ((cumsum) <= tau).sum(axis=1) + 1  # 1 - 1001
    sizes_base = np.minimum(sizes_base, scores.shape[1]) # 1-1000

    if randomized:
        V = np.zeros(sizes_base.shape)
        for i in range(sizes_base.shape[0]):
            V[i] = 1/ordered[i,sizes_base[i]-1] * \
                    (tau-(cumsum[i,sizes_base[i]-1]-ordered[i,sizes_base[i]-1])) # -1 since sizes_base \in {1,...,1000}.

        sizes = sizes_base - (np.random.random(V.shape) >= V).astype(int)
    else:
        sizes = sizes_base

    if tau == 1.0:
        sizes[:] = cumsum.shape[1] # always predict max size if alpha==0. (Avoids numerical error.)

    if not allow_zero_sets:
        sizes[sizes == 0] = 1 # allow the user the option to never have empty sets (will lead to incorrect coverage if 1-alpha < model's top-1 accuracy

    S = list()

    # Construct S from equation (5)
    for i in range(I.shape[0]):
        S = S + [I[i,0:sizes[i]],]

    return S


def get_tau(score, target, I, ordered, cumsum, randomized, allow_zero_sets): # For one example
    idx = np.where(I==target)
    tau_nonrandom = cumsum[idx]

    if not randomized:
        return tau_nonrandom
    
    U = np.random.random()

    if idx == (0,0):
        if not allow_zero_sets:
            return tau_nonrandom
        else:
            return U * tau_nonrandom
    else:
        return U * ordered[idx] + cumsum[(idx[0],idx[1]-1)]


def calib(logits, labels, alpha=0.1, batch_size=64, plat_max_iters=10, plat_lr=0.01, randomized=True, allow_zero_sets=True):
    dataloader = get_logits_dataloader(logits, labels, batch_size=batch_size, shuffle=False, pin_memory=True)
    t = platt_logits(dataloader, max_iters=plat_max_iters, lr=plat_lr, epsilon=0.01)
    qhat = conformal_calibration_logits(dataloader, t, alpha, randomized, allow_zero_sets)
    return {'qhat': qhat, 't': t}


def predict_sets(logits, qhat, t=1., randomized=True, allow_zero_sets=True):
    probs = softmax(logits / t , 1)
    I, ordered, cumsum = sort_sum(probs)
    sets = gcq(probs, qhat, I=I, ordered=ordered, cumsum=cumsum, randomized=randomized, allow_zero_sets=allow_zero_sets)
    return sets


def eval_sets(sets, labels, print_bool=True):
    total = 0
    correct = 0
    set_size = []
    for s, l in zip(sets, labels):
        total += 1
        if l in s:
            correct += 1
        set_size.append(len(s))
    acc = correct / float(total)
    set_size = np.asarray(set_size)
    if print_bool:
        print('Acc: {:.4f}'.format(acc))
        print('Mean set sizes: {:.2f} ({:.2f})'.format(set_size.mean(), set_size.std()))
    else:
        return {'acc': acc, 'size': set_size.mean()}


if __name__ == '__main__':
    from evaluate import load_pickle, split_data
    
    par_test = 0.2
    alpha = 0.1
    file_name = '/home/royhirsch/conformal/data/embeds_n_logits/imnet1k_r152/valid.pickle'

    all_mets = {'acc': [], 'size': []}
    data = load_pickle(file_name)
    for i in range(20):
        valid_data, calib_data = split_data(data, par_test=par_test, seed=i)
        calib_out = calib(calib_data['preds'], calib_data['labels'], alpha)
        valid_sets = predict_sets(valid_data['preds'], calib_out['qhat'], calib_out['t'])
        mets = eval_sets(valid_sets, valid_data['labels'], print_bool=False)
        all_mets['acc'].append(mets['acc'])
        all_mets['size'].append(mets['size'])
    print('Acc: {:.4f}'.format(np.asarray(all_mets['acc']).mean()))
    print('Mean set sizes: {:.2f}'.format(np.asarray(all_mets['size']).mean()))
    print('Median set sizes: {:.2f}'.format(np.median(np.asarray(all_mets['size']))))


    
