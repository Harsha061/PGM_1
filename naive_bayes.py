import numpy as np
from numpy import union1d, intersect1d
import pdb


class NBayes:

    def __init__(self,laplacian=1):
        self.laplacian = laplacian
    
    def fit(self,features, y):
        self.features = np.array(features)
        self.y = np.array(y)
        self.num_features = features.shape[1]
        self.classes = np.unique(y)
        self.unique_features = [np.unique(self.features[:,i]) for i in range(self.num_features)]
        self.probabs = []
        self.class_idx = {}
        self.priors = {}

        for i in self.classes:
            self.class_idx[i] = np.where(self.y==i)[0]
            self.priors[i] = (len(self.class_idx[i])+self.laplacian)/(len(self.y) + self.laplacian*len(self.classes))            

        for i in range(self.num_features):
            feat_probabs = {}
            for x in self.unique_features[i]:
                idx1 = np.where(self.features[:,i]==x)[0]
                class_probabs = {}
                for c in self.classes:
                    idx2 = self.class_idx[c]
                    pr = (len(intersect1d(idx1,idx2))+self.laplacian)/(len(idx2)+self.laplacian*len(self.unique_features[i]))
                    class_probabs[c] = pr
                feat_probabs[x] = class_probabs
            self.probabs.append(feat_probabs)
    
    def infer_logprobab(self,feature, y):
        ans = np.log(self.priors[y])
        for f in range(self.num_features):
            ans+= np.log(self.probabs[f][feature[f]][y])
        return ans
    
    def predict(self, features):
        y_pred = []
        for i in range(features.shape[0]):
            probabs = [self.infer_logprobab(features[i], y) for y in self.classes]
            y = self.classes[np.argmax(np.array(probabs))]
            y_pred.append(y)
        return np.array(y_pred)

if __name__ == "__main__":
    laplacians = [1,0.5,0.1,2]
    from utils import read_csv, create_csv
    import utils

    for l in laplacians:
        model = NBayes(l)
        train_data = read_csv(utils.TRAIN_PATH)
        x_train = train_data[:,1:]
        y_train = train_data[:,0]

        model.fit(x_train, y_train)

        test_data = read_csv(utils.TEST_PATH)[:,1:]
        y_pred = model.predict(test_data)

        create_csv(y_pred, 'submissions/NBayes_'+str(l)+'.csv')

