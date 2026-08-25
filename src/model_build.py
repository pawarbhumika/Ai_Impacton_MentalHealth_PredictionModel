from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report

def model(X_train,X_test,y_train,y_test):
    model_rf=RandomForestClassifier(random_state=1)
    
    model_rf.fit(X_train,y_train)
    
    y_pred=model_rf.predict(X_test)
    
    a=accuracy_score(y_test,y_pred)
    print('accuracy',a)
    
    print(classification_report(y_test,y_pred))
    
    return y_pred,model_rf
    

