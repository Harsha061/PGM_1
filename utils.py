import pandas as pd
import numpy as np

TEST_PATH = 'data/test.csv'
TRAIN_PATH = 'data/train.csv'

def read_csv(path):
    return pd.read_csv(path).to_numpy()
    
def create_csv(ypred, path):
    ypred = np.array(ypred).astype(np.int)
    df = pd.DataFrame({'id':np.arange(len(ypred))+1, 'class':ypred})
    df.to_csv(path,index=False)
