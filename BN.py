import numpy as np
from numpy import union1d, intersect1d
from itertools import product
import pdb

class BayesN:

    def __init__(self, nodes, edges, laplacian=1):
        self.n = nodes
        self.edges = edges
        self.parents = [set() for i in range(nodes)]
        for e in edges:
            self.parents[e[1]].add(e[0])
        self.parents = [list(x) for x in self.parents]
        
        self.laplacian = laplacian
    
    def fit(self, data):
        data = np.array(data)
        assert data.shape[1]==self.n, 'Incorrect shape:'+str(data.shape)

        self.unique = [np.unique(data[:,i]) for i in range(self.n)]
        self.feat_len = [len(i) for i in self.unique]

        self.probabs = []

        for f in range(self.n):
            if len(self.parents[f]) == 0:
                prb = np.ones(self.feat_len[f])/self.feat_len[f]
            else:
                parent_values = [self.unique[i] for i in self.parents[f]]
                parent_val_size = [self.feat_len[i] for i in self.parents[f]]
                prb = np.zeros((self.feat_len[f],)+tuple(parent_val_size))
                for v in self.unique[f]:
                    for x in product(*parent_values):
                        idx = np.arange(data.shape[0])
                        for p in enumerate(self.parents[f]):
                            idx = intersect1d(idx,np.where(data[:,p[1]]==x[p[0]])[0])
                        idx2 = intersect1d(idx, np.where(data[:,f]==v)[0])
                        val = (len(idx2)+self.laplacian)/(len(idx)+self.laplacian*self.feat_len[f])
                        prb.itemset((v,)+x,val)
            self.probabs.append(prb)
        
    def get_joint_log(self, features):
        ans = 0
        for f in range(self.n):
            parent_vals = [features[i] for i in self.parents[f]]
            ans += np.log(self.probabs[f].item(*((features[f],)+ tuple(parent_vals))))
        return ans
    
    def predict(self, x_vals):
        assert x_vals.shape[1]==self.n-1, 'Incorrect shape:'+str(x_vals.shape)
        y_pred = []
        for i in range(x_vals.shape[0]):
            probs = []
            for j in self.unique[-1]:
                l = list(x_vals[i])
                l.append(j)
                probs.append(self.get_joint_log(l))
            y = self.unique[-1][np.argmax(probs)]
            y_pred.append(y)
        return y_pred
            
if __name__ == "__main__":
    from utils import read_csv, create_csv
    import utils
    nodes = 23
    edges = [(22,i) for i in range(22)]
    edges.append((15,7))
    edges.append((15,8))
    model = BayesN(nodes=nodes, edges=edges)
    train_data = read_csv(utils.TRAIN_PATH)
    x_train = train_data[:,1:]
    y_train = train_data[:,0]
    train_data = np.concatenate((x_train,y_train.reshape(y_train.shape+(1,))), axis=1)
    test_data = read_csv(utils.TEST_PATH)[:,1:]
    laplacians = [1,0.5,0.1,2]
    for l in laplacians:
        model = BayesN(nodes, edges,l)
        model.fit(train_data)
        y_pred = model.predict(test_data)
        create_csv(y_pred, 'submissions/NBayes1_'+str(l)+'.csv')
