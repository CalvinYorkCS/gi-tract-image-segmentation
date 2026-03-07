import pandas as pd
from sklearn.model_selection import KFold

if __name__ == "__main__":
    df = pd.read_csv("../input/train.csv")
    df["kfold"] = -1
    df.sample(frac=1).reset_index(drop=True)
    
    kf = KFold(n_splits=5)
    
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold
    
    df.to_csv("../input/train_folds.csv", index=False)