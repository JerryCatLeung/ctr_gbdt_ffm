import data_provider
import gbdt_ffm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, log_loss, accuracy_score
from sklearn.metrics.ranking import roc_auc_score
import code

def simple_feature_eng(X_train, X_test):
    
    X = np.vstack((X_train, X_test))
    df_X = pd.DataFrame(X)
    
    int_cols = [i for i in range(0, 13)]
    cat_cols = [i for i in range(13, 39)]

    num_of_cate = {}
    for col in cat_cols:
        unique = set(df_X[col].tolist())
        num_of_cate[col] = len(unique)

    cats = []
    for col in num_of_cate:
        if num_of_cate[col] > 200:
            df_X = df_X.drop(col, 1)
        else: 
            cats += [col]

    for col in cats:
        df_X = pd.concat([df_X, pd.get_dummies(df_X[col], prefix = col)], axis=1)
        df_X = df_X.drop(col, axis=1)

    X_train_new = df_X[: X_train.shape[0]].as_matrix()
    X_test_new = df_X[X_train.shape[0]: ].as_matrix()

    return X_train_new, X_test_new
    

def lr(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression
    X_train, X_test = simple_feature_eng(X_train, X_test)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_test_pred = model.predict_proba(X_test)
    return y_test_pred


def knn(X_train, X_test, y_train, y_test):
    from sklearn.neighbors import KNeighborsClassifier
    X_train, X_test = simple_feature_eng(X_train, X_test)
    model = KNeighborsClassifier(n_neighbors=30)
    model.fit(X_train, y_train)
    y_test_pred = model.predict_proba(X_test)
    return y_test_pred
  
 
def rf(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    X_train, X_test = simple_feature_eng(X_train, X_test)
    rf_model = RandomForestClassifier(n_estimators=30, max_depth=10)
    rf_model.fit(X_train, y_train)
    y_test_pred = rf_model.predict_proba(X_test)
    return y_test_pred


def adb(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    X_train, X_test = simple_feature_eng(X_train, X_test)
    model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=20), algorithm="SAMME", n_estimators=20)
    model.fit(X_train, y_train)
    y_test_pred = model.predict_proba(X_test)
    return y_test_pred


def gbdt(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import GradientBoostingClassifier
    X_train, X_test = simple_feature_eng(X_train, X_test)
    model = GradientBoostingClassifier(learning_rate=0.1, max_depth=4, loss='deviance')
    model.fit(X_train, y_train)
    y_test_pred = model.predict_proba(X_test)
    return y_test_pred

 
def plot_roc(name, y_test, y_pred):
    n = y_pred.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    y_test = y_test.reshape(y_test.shape[0], 1)
    y_test = np.hstack((1 - y_test, y_test))
    
    for i in range(n):
        fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_pred[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.plot(fpr[1], tpr[1], alpha=0.8, label=name + ' (area = %0.3f)' % roc_auc[1])
    plt.legend(loc='best')


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = data_provider.get_criteo_data("./train_sample.txt")
    
    # Logistic Regression
    y_test_pred_lr = lr(X_train, X_test, y_train, y_test)

    # Knn
    y_test_pred_knn = knn(X_train, X_test, y_train, y_test)
    
    # Random Forest
    y_test_pred_rf = rf(X_train, X_test, y_train, y_test)

    # Adaboost
    y_test_pred_adb = adb(X_train, X_test, y_train, y_test)

    # gbdt
    y_test_pred_gbdt = gbdt(X_train, X_test, y_train, y_test)
    
    # gbdt + ffm
    y_test_pred_gbdtffm = gbdt_ffm.perform(X_train, X_test, y_train, y_test)
    

    # evalution

    # logloss
    lr_loss = log_loss(y_test, y_test_pred_lr)
    knn_loss = log_loss(y_test, y_test_pred_knn)
    rf_loss = log_loss(y_test, y_test_pred_rf)
    adb_loss = log_loss(y_test, y_test_pred_adb)
    gbdt_loss = log_loss(y_test, y_test_pred_gbdt)
    gbdtffm_loss = log_loss(y_test, y_test_pred_gbdtffm)

    print('lr loss: ', lr_loss)  
    print('knn loss: ', knn_loss)  
    print('rf loss: ', rf_loss) 
    print('adaboost loss: ', adb_loss)  
    print('gbdt loss: ', gbdt_loss)  
    print('gbdtffm loss: ', gbdtffm_loss) 

    
    # plot roc curve
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])        
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves')
    plot_roc('lr', y_test, y_test_pred_lr)
    plot_roc('knn', y_test, y_test_pred_knn)
    plot_roc('rf', y_test, y_test_pred_rf)
    plot_roc('adaboost',  y_test, y_test_pred_adb)
    plot_roc('gbdt',  y_test, y_test_pred_gbdt)
    plot_roc('gbdtffm', y_test, y_test_pred_gbdtffm)
    plt.savefig('./roc-auc.png')
    plt.show()

