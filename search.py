from BN import BayesN
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from utils import read_csv, create_csv
import utils
import pdb

def get_model_accuracy(model, train, test, y):
    model.fit(train)
    y_pred = model.predict(test)
    return accuracy_score(y,y_pred)

def get_train():
    train_data = read_csv(utils.TRAIN_PATH)
    x_train = train_data[:,1:]
    y_train = train_data[:,0]
    train_data = np.concatenate((x_train,y_train.reshape(y_train.shape+(1,))), axis=1)
    return train_data

def get_kfold_accuracy(model, data, splits=3, seed=0):
    kf = KFold(n_splits=splits, shuffle=True, random_state=seed)
    acc = []
    for train_idx, test_idx in kf.split(data):
        test, train = data[test_idx], data[train_idx]
        x_test, y_test = test[:,:-1], test[:,-1]
        acc.append(get_model_accuracy(model, train, x_test, y_test))
    return np.mean(acc)

def greedy_learn(steps=5, seed=None, splits=10, save=True, savepath = 'submissions/greedy.csv', failed_runs=10000):
    rg = np.random.RandomState(seed)
    toposort = list(rg.permutation(22))
    extra_edges = set()
    nodes = 23
    edges = [(22,i) for i in range(22)]
    train = get_train()
    max_score = get_kfold_accuracy(BayesN(nodes=nodes, edges=edges), train, splits)
    #print('Step 0:', max_score)
    st = 0
    f = 0
    while st < steps and f < failed_runs:
        while True:
            u = rg.randint(22)
            v = rg.randint(22)
            if u == v: continue
            cand_edge = (u,v) if toposort.index(u)<toposort.index(v) else (v,u)
            break
        edges.append(cand_edge)
        score = get_kfold_accuracy(BayesN(nodes=nodes, edges=edges), train, splits)
        if score > max_score:
            extra_edges.add(cand_edge)
            max_score = score
            st += 1
            #print('Step',st,':',max_score)
        else:
            edges.pop()
            f+= 1
    print('Max Score:', max_score)
    if save:
        test_data = read_csv(utils.TEST_PATH)[:,1:]
        model = BayesN(nodes=nodes, edges=edges)
        model.fit(train)
        y_pred = model.predict(test_data)
        create_csv(y_pred,savepath)
    return model

if __name__ == "__main__":
    seeds = list(range(100))
    for s in seeds:
        print('Seed:',s)
        m = greedy_learn(steps=30, seed=s, savepath='submissions/greedy_'+str(s)+'.csv')


