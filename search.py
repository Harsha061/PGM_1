from BN import BayesN
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from utils import read_csv, create_csv
import utils
import random, copy
import networkx as nx
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

    #Initial Edges
    edges = [(22,i) for i in range(22)]
    train = get_train()
    #Get score on naive Bayes
    max_score = get_kfold_accuracy(BayesN(nodes=nodes, edges=edges), train, splits)

    st = 0
    f = 0
    while st < steps and f < failed_runs:
        while True:
            #Select Edges
            u = rg.randint(22)
            v = rg.randint(22)
            if u == v or (u,v) in edges or (v,u) in edges: continue
            cand_edge = (u,v) if toposort.index(u)<toposort.index(v) else (v,u)
            break
        edges.append(cand_edge)

        #Get Score with new network
        score = get_kfold_accuracy(BayesN(nodes=nodes, edges=edges), train, splits)

        #Check if addtion of edge increased performance
        if score > max_score:
            extra_edges.add(cand_edge)
            max_score = score
            st += 1
            print('Step',st,':',max_score,'Extra Edges:',extra_edges)
        else:
            edges.pop()
            f+= 1
    print('Max Score:', max_score)
    print('Extra Edges:', extra_edges)
    if save:
        test_data = read_csv(utils.TEST_PATH)[:,1:]
        model = BayesN(nodes=nodes, edges=edges)
        model.fit(train)
        y_pred = model.predict(test_data)
        create_csv(y_pred,savepath)
    return model

def greedy_learn2(steps=5, seed=None, splits=10, save=True, savepath = 'submissions/greedy.csv', failed_runs=10000):
    rg = np.random.RandomState(seed)
    g = nx.DiGraph()
    for i in range(22):
        g.add_edge(22,i)
    nodes = 23

    #initialize edges as Naive Bayes
    edges = set([(22,i) for i in range(22)])
    train = get_train()

    #Get score for Naive Bayes
    max_score = get_kfold_accuracy(BayesN(nodes=nodes, edges=edges), train, splits)
    print('Step 0:', max_score)
    st = 0
    f = 0
    while st < steps and f < failed_runs:
        #Select r
        r = rg.randint(3)
        gr = copy.deepcopy(g)
        ed = copy.deepcopy(edges)

        #Add random edge
        if r == 0:
            while True:
                u = rg.randint(22)
                v = rg.randint(22)
                if u==v or (u,v) in g.edges: continue
                g.add_edge(u,v)
                #Reject if G in not DAG
                if not nx.is_directed_acyclic_graph(g):
                    g.remove_edge(u,v)
                    continue
                break
            if nx.is_directed_acyclic_graph(g):
                edges.add((u,v))
        #Delete a random edge
        elif r == 1:
            if len(edges)<=3: continue
            del_edge = random.sample(edges,1)[0]
            edges.remove(del_edge)
            g.remove_edge(*del_edge)
        
        #Reverse orientation of random edge
        else:
            if len(edges)<=3: continue
            for i in range(len(edges)):
                act_edge = random.sample(edges,1)[0]
                rev_edge = (act_edge[1], act_edge[0])
                g.remove_edge(*act_edge)
                g.add_edge(*rev_edge)
                if nx.is_directed_acyclic_graph(g): 
                    break
                else:
                    g.remove_edge(*rev_edge)
                    g.add_edge(*act_edge)
            if nx.is_directed_acyclic_graph(g):
                edges.remove(act_edge)
                edges.add(rev_edge)
                
        #Get Score on new BN
        score = get_kfold_accuracy(BayesN(nodes=nodes, edges=edges), train, splits)
        if score > max_score:
            max_score = score
            st += 1
            print('Step',st,':',max_score, 'Edges:', edges)
        else:
            edges = ed
            g = gr
            f+= 1
    print('Max Score:', max_score)
    print('Edges:', edges)
    if save:
        test_data = read_csv(utils.TEST_PATH)[:,1:]
        model = BayesN(nodes=nodes, edges=edges)
        model.fit(train)
        y_pred = model.predict(test_data)
        create_csv(y_pred,savepath)
    return model


if __name__ == "__main__":
    seeds = list(range(10))
    models = []
    for s in seeds:
        print('Seed:',s)
        m = greedy_learn2(steps=20, seed=s,splits=5, savepath='submissions/greedyd__'+str(s)+'.csv', failed_runs=5000)
        models.append(m)


