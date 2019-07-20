import numpy as np  # numpy库
import pandas as pd  # 导入pandas
import os
import pickle

##############################################################################################################
## 调用各组织的模型
def svrmodel(mat,Filename):
    ## 调用模型
    sample=mat.columns.values.tolist()
    X=np.array(mat).T
    path = Dir + '/'+t+'_SVR.pickle'
    with open(path, 'rb') as f:
        model = pickle.load(f)
        y_pre = model.predict(X)
        f.close()
    df=pd.DataFrame(y_pre, index=sample, columns=['DNAmAge'])
    print(df)
    ##############################################################################################################
    ## 模型结果保存
    Out=Dir + '/'+Filename+'_'+t+'_DNAmAge.csv'
    df.to_csv(Out)
if __name__ == "__main__":
    Dir=os.getcwd()
    tissue_id=[
        'buccal',
        'breast',
        'saliva',
        'brain',
        'blood'
    ]
    for t in tissue_id:
        Filename= 'Filename'
        mat=pd.read_csv(Filename+'.csv',encoding = "utf-8",index_col=0,header=0)
        svrmodel(mat,Filename)
