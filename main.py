from src.data_ingestion import data_loader
from src.data_preprocessing import preprocessing
from src.model_build import model

def main():
    df=data_loader()
    print(df.shape)

    X_train,X_test,y_train,y_test,transform=preprocessing(df)
    print(y_train.value_counts())


    score ,feature_importance=model(X_train,X_test,y_train,y_test)


  
if __name__ == "__main__":
    main()