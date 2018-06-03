import data_provider
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.ranking import roc_auc_score
from sklearn.preprocessing.data import OneHotEncoder
from sklearn.metrics import log_loss, accuracy_score
import math
from fastFM import als, sgd
import xlearn as xl
import hashlib
import cPickle
import csv
import code
import os

def hashstr(str):
    return int(hashlib.md5(str.encode('utf8')).hexdigest(), 16) % (1e+6 - 1) + 1

def gen_hashed_fm_feats(feats):
    feats = ['{0}:{1}:1'.format(field - 1, hashstr(feat)) for (field, feat) in feats]
    return feats

def gen_ffm_feature(X_train, gbdt_code, stat_dict):
    frequent_feats = stat_dict['frequent_feats']

    features = []
    for row in range(len(X_train)):
        feature = []
        # X_train int
        for i in range(0, 13):
            field = "I" + str(i + 1)
            content = field + "-"
            if X_train[row][i] != "nan":
                value = int(float(X_train[row][i]))
                if value > 2:
                    content = field + "-" + str(int(math.log(float(value)) ** 2))
                else:
                    content = field + "-" + "SP" + str(value)
            feature.append(content)

        # X_train cate
        for i in range(0, 26):
            field = "C" + str(i + 1)
            content = ''
            feat = X_train[row][13 + i]
            if (field + "-" + feat) not in frequent_feats:
                content = field + "x"
            else:
                 content = field + "-" + feat
            feature.append(content)

        # gbdt_code
        for i in range(len(gbdt_code[row])):
            feature.append(str(i + 1) + ":" + str(int(gbdt_code[row][i])))

        features_dict = [(i + 1, f) for i, f in enumerate(feature)]
        hashed_features = gen_hashed_fm_feats(features_dict)

        features.append(hashed_features) 
        
    return features


def dump_ffm_features(f_name, ffm_features, labels):
    f = open(f_name, 'w')
    for row in range(len(ffm_features)):
        f.write(str(labels[row]) + ' ')
        for col in range(len(ffm_features[0])):
            f.write(ffm_features[row][col] + ' ')
        f.write('\n')


def dump_to_xlearn(X_train, X_test, y_train, y_test):
    a = y_train.reshape(y_train.shape[0], 1)
    train = np.hstack((a, X_train.toarray()))

    b = y_test.reshape(y_test.shape[0], 1)
    test = np.hstack((b, X_test.toarray()))
    
    np.savetxt('./train_xlearn.csv', train, delimiter='\t', fmt='%f')
    np.savetxt('./test_xlearn.csv', test, delimiter='\t', fmt='%f')


def read_pred(f_name):
    data = np.loadtxt(f_name)
    return data


def stack_features(dense, sparse):
    # my_data1 = 
    # my_data2 = np.zeros((len(lines), 26))
    
    for i, line in enumerate(lines):
        tokens = line.split(" ")
        for token in tokens[1: ]:
            num = int(token)
            my_data2[i][num - 1] = 1

    label = my_data1[:, 0]
    my_data1 = my_data1[:, 1:]
    my_data = np.hstack((my_data1, my_data2))

    return my_data, label


def feat_eng(X, stat_dict):
    feature_list = stat_dict['feature_list']

    X_dense = X[:, 0: 13]
    X_sparse = []
    for row in range(X.shape[0]):
        cur_feats = set()
        for col in range(0, 26):
            t = str(X[row][13 + col])
            if t == 'nan':
                t = ''
            cur_feats.add('C' + str(col + 1) + '-' + t)

        tmp = []
        for i, f in enumerate(feature_list, start=1):
            if f in cur_feats:
                tmp.append(str(i))
        X_sparse.append(tmp)

    my_data1 = X_dense
    my_data2 = np.zeros((len(X_sparse), 26))

    for i, tokens in enumerate(X_sparse):
        for token in tokens:
            num = int(token)
            my_data2[i][num - 1] = 1

    my_data = np.hstack((my_data1, my_data2))

    return my_data.astype(float)


def clean_tmp_files(tmp_folder_name):
    os.system("rm -rf " + tmp_folder_name)


def perform(X_train_src, X_test_src, y_train, y_test):
    tmp_folder_name = './tmp'
    if not os.path.exists(tmp_folder_name):
        os.makedirs(tmp_folder_name)

    in_file = open('stat.pkl', 'rb')
    stat_dict = cPickle.load(in_file)
    in_file.close()

    X_train_gbdt = feat_eng(X_train_src, stat_dict)
    X_test_gbdt = feat_eng(X_test_src, stat_dict)
    
    X_train_gbdt_sp = sparse.csr_matrix(X_train_gbdt)
    X_test_gbdt_sp = sparse.csr_matrix(X_test_gbdt)


    model_gbdt = GradientBoostingClassifier(
                n_estimators=30, 
                learning_rate=0.05,
                max_depth=7, 
                verbose=0,
                max_features=0.6)

    model_gbdt.fit(X_train_gbdt_sp, y_train)

    gbdt_y_pred = model_gbdt.predict(X_test_gbdt)
    gbdt_y_predprob = model_gbdt.predict_proba(X_test_gbdt)[:, 1]
    
    gbdt_acc = accuracy_score(y_test, gbdt_y_pred)
    gbdt_auc = roc_auc_score(y_test, gbdt_y_predprob)
    gbdt_loss = log_loss(y_test, gbdt_y_predprob)

    print('gbdt accuracy : %.3g' % gbdt_acc)
    print('gbdt auc: %.3f' % gbdt_auc)
    print('gbdt loss: %.3f' % gbdt_loss)

    gbdt_train_code = model_gbdt.apply(X_train_gbdt_sp)[:, :, 0]
    gbdt_test_code = model_gbdt.apply(X_test_gbdt_sp)[:, :, 0]

    X_train_src = X_train_src.astype(str).tolist()
    X_test_src = X_test_src.astype(str).tolist()

    ffmfeatures_train = gen_ffm_feature(X_train_src, gbdt_train_code, stat_dict)
    ffmfeatures_test = gen_ffm_feature(X_test_src, gbdt_test_code, stat_dict)
  

    dump_ffm_features(tmp_folder_name + '/tmp_tr.ffm', ffmfeatures_train, y_train)
    dump_ffm_features(tmp_folder_name + '/tmp_te.ffm', ffmfeatures_test, y_test)

    ffm_model = xl.create_ffm()
    ffm_model.setTrain(tmp_folder_name + "/tmp_tr.ffm")
    ffm_model.setValidate(tmp_folder_name + "/tmp_te.ffm")

    param = {
        'task':'binary', \
        'lr':0.2, \
        'lambda':0.002, \
        'opt': 'sgd', \
        'epoch': 10
    }

    ffm_model.fit(param, tmp_folder_name + "/ffm_model.out")
    
    # ffm_model.cv(param)

    ffm_model.setTest(tmp_folder_name + "/tmp_te.ffm")
    ffm_model.setSigmoid()
    y_test_pred = ffm_model.predict(tmp_folder_name + "/ffm_model.out", tmp_folder_name + "/ffm_output.txt")
    
    y_test_pred = read_pred(tmp_folder_name + "/ffm_output.txt")

    # test_loss = log_loss(y_test, y_pred_test)
    # print('shw test loss: ', test_loss) 
    
    clean_tmp_files(tmp_folder_name)

    tmp = y_test_pred.reshape(y_test_pred.shape[0], 1)
    y_test_pred = np.hstack((1 - tmp, tmp))

    return y_test_pred

    

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = data_provider.get_criteo_data("./train_sample.txt")
    y_test_pred = perform(X_train, X_test, y_train, y_test)

    code.interact(local=locals())



    
