import pandas as pd
import numpy as np
import argparse
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
def load_data(path):
    data_df = pd.read_csv(path)
    data_df = data_df.fillna(data_df.median())
    target_data = data_df['Target']
    data_df = data_df.drop('Target',axis=1)
    data = np.array(data_df)
    target_data = np.array(target_data).reshape(-1,1)
    return data,target_data,path

def process_type(choice):
    if(choice ==0):
        return None
    elif(choice == 1):
        return MinMaxScaler()
    else:
        return StandardScaler()
def process_data(train, test,choice):
    processor = process_type(choice)
    if processor is not None:
        train = processor.fit_transform(train)
        test = processor.transform(test)
    return train,test

def get_r2(train,target):
    return r2_score(train,target)
def evaluate(model,train,target_train,test,target_test):
    pred_train = regr.predict(train)
    pred_test = regr.predict(test)
    train_r2 = get_r2(pred_train,target_train)
    test_r2 = get_r2(pred_test, target_test)
    print(f'Train R2 is : {train_r2}, Test R2 is : {test_r2}')
    return 0

if __name__ == '__main__':
    
    path = 'data2_200x30.csv'

    parser = argparse.ArgumentParser(description='Training a simple Neural Network using SKlearn.')
    parser.add_argument('--dataset', type=str,default=path)
    parser.add_argument('--preprocessing', type=int,default=1, help='0 for nothing, 1 for min/max scaling, 2 for standarizing.')
    args = parser.parse_args()

    data,target_data,path = load_data(args.dataset)

    train, test, target_train,target_test = train_test_split(data,target_data,test_size=.2,shuffle=True,random_state=5)
    preprocessor = args.preprocessing
    
    train,test = process_data(train,test,args.preprocessing)
    ## Modeling

    regr = MLPRegressor(hidden_layer_sizes=(5,5,5), random_state=5,max_iter=10000)
    regr.fit(train,target_train)
    pred_train = regr.predict(train)
    pred_test = regr.predict(test)

    ## Evaluate

    evaluate(regr, train,target_train,test,target_test)
