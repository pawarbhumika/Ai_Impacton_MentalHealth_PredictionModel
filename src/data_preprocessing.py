from sklearn.preprocessing import MinMaxScaler,RobustScaler,OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer

def preprocessing(df):
    
    df.drop_duplicates(inplace = True)
    
    
    X=df.drop(columns=['Risk_Category'])
    y=df['Risk_Category']   
    

    
    X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                     test_size=0.3,
                                                     random_state=1)
    
    numerical_column = X.select_dtypes(exclude = 'object').columns
    categorical_column = X.select_dtypes(include = 'object').columns
    
    numerical_pipeline = Pipeline(steps = [('imputer',SimpleImputer(strategy='mean')),
                                           ('Scaler',MinMaxScaler())])
    categorical_pipeline = Pipeline(steps =[('imputer',SimpleImputer(strategy='most_frequent')),
                                            ('Encoder', OneHotEncoder(handle_unknown='ignore',drop='first'))])
    
    transformer = ColumnTransformer(transformers =[('num',numerical_pipeline,numerical_column),
                                                   ('cat',categorical_pipeline,categorical_column)])   
    
    X_train = transformer.fit_transform(X_train)
    X_test = transformer.transform(X_test)
    
    sm = SMOTE(random_state = 1)
    X_train,y_train = sm.fit_resample(X_train,y_train)
    
    pca=PCA(n_components=0.95)
    X_train=pca.fit_transform(X_train)
    X_test=pca.transform(X_test)
    
    
    return X_train,X_test,y_train,y_test,transformer




