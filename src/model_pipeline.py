"""
Model Training and Prediction Module for Spaceship Titanic Classification

This module handles the complete ML pipeline including:
 • Train/validation/test splitting
 • Feature preprocessing pipelines
 • Model training with XGBoost
 • Model evaluation
 • Prediction on test data

Author: Priyank
Date: January 2026
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy  as np
import pickle




class ModelPipeline():
    """
    A complete ML pipeline for Spaceship Titanic passenger transport prediciton.

    This class handles:
    • Data splitting 
    • Feature preprocessing (Scaling, Encoding, imputation)
    • Model training with XGBoost
    • Model evaluation with multiple metrics
    • Prediction and submission file generation

    Attributes:
        train_dir (str): Path to training data csv
        test_dir (str): Path to test data csv

    """
    
    def __init__(self, dir_train = r"./data/raw/train.csv", 
                 dir_test = r"./data/raw/test.csv"):
        
        
        self.train_dir = dir_train
        self.test_dir = dir_test

    def train_test_val_split_dx(self, dir_: str) -> pd.DataFrame:
        """
        Split data into training, validation, and test sets with stratification.

        Create a 70-15-15 split:
        • 70% for training
        • 15% for validation
        • 15% for final test evaluation

        Args:
            dir_ (str): Path to the data file
        Returns:
            DataFrames
        """
        print("Getting Data....")
        dx  = pd.read_csv(dir_)

        dx["Transported"] = dx["Transported"].astype("Int64")
        
        X = dx.drop(columns=["Transported"]).copy()
        y = dx["Transported"]
        
        X_trainfull, X_temp, y_trainfull, y_temp = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp,random_state=42
        )
        
        return X_trainfull, y_trainfull, X_val, y_val, X_test, y_test

    def get(self, X: pd.DataFrame,  is_train: bool= True)-> pd.DataFrame:

        """
        Engineer features from raw passenger data.

        Feature engineering steps:
        1. Extract group and group size from passengerId
        2. Parse cabin into Deck, Side, Num
        3. Create TotalSpend from spending columns
        4. Impute CryoSleep based on spending patterns
        5. Log-Transform spending feature to normalize the distribution
        
        Args: 
            X (pd.DataFrame): Raw features
            is_train boolean: True/False

        Returns:
            pd.DataFrame: Engineered features
            If is_train=False, returns (features, passenger_id)
        """
        
        id_ = X['PassengerId']
        X['Group'] = X['PassengerId'].str.split("_").str[0]
        X['GroupSize'] = X.groupby("Group")['Group'].transform('count')

        X = X.drop(columns= ["PassengerId", "Name", "Group"])
        
        
        spend_col = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

        X[['Deck', 'Num', 'Side']] = X['Cabin'].str.split("/", expand = True)

        X['Num'] = X['Num'].astype("Int64")
       

        X.drop(columns=['Cabin'], inplace =True)

        X["TotalSpend"] = X[spend_col].sum(axis =1)


        X.loc[(X['CryoSleep'].isna()) & (X['TotalSpend']>0), 'CryoSleep'] = False
        X.loc[(X['CryoSleep'].isna()) & (X['TotalSpend']==0), 'CryoSleep'] = True

        obj_cols = X.select_dtypes(include='object').columns

        for col in spend_col:
        
            X[f"{col}_log"] = np.where(
                X[col] > 0,
                np.log1p(X[col]),
                0
            )
            
            X[f"{col}_used"] = (X[col] > 0).astype(int)
            X.drop(columns=[col], inplace=True)
        
        for col in obj_cols:
            X[col] = X[col].astype(object)

        obj_cols = X.select_dtypes(include=['object', "bool"]).columns

        for col in obj_cols:
            X[col] = X[col].astype(object)

        X = X[['HomePlanet', 'CryoSleep', 'GroupSize','Destination', 'Age',  'Deck', 'Num', 'Side', 'TotalSpend', 'RoomService_log', 
                'FoodCourt_log',  'ShoppingMall_log',  'Spa_log', 
                'VRDeck_log', ]]
        
        if not is_train:
            return X, id_
        
        return X
    
    def preprocess_pipeline(self, X: pd.DataFrame,
                            X_val: pd.DataFrame,
                            y: pd.Series, 
                            y_val: pd.Series) -> Pipeline:
        """
        Create and train preprocessing + model pipeline.

        Pipeline components:
        1. Numeric features: Median imputaion + RobustScaler
        2. Categorical features: Model imputation + OneHotEncoding
        3. XGBoost classifier with early stopping on validation set

        Args: 
            X (pd.DataFrame): Training features
            X_val (pd.DataFrame): Validation features
            y (pd.Series): Training target
            y_val (pd.Series): Validataion target

        Returns:
            Pipeline: Fitted sklearn Pipeline object
        """

        num_cols = X.select_dtypes(include=['int64','float64']).columns
        cat_cols = X.select_dtypes(include=['object','category']).columns

        
        
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", RobustScaler())
                    ]),
                    num_cols
                ),
                (
                    "cat",
                    Pipeline(steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                    ]),
                    cat_cols
                )
            ],
            remainder="drop"
        )

        model = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("classifier", XGBClassifier(
                    n_estimators=1000,
                    max_depth = 5,
                    learning_rate = 0.05,
                    subsample=0.8,
                    colsample_bytree= 0.8,
                    random_state=42,
                    use_label_encoder = False,
                    eval_metric="logloss",
                    early_stopping_rounds = 50, 
                ))
            ])
        
        X_transformed = model.named_steps['preprocess'].fit_transform(X)
        X_val_transformed = model.named_steps['preprocess'].transform(X_val)
        
        model.named_steps['classifier'].fit(
            X_transformed, y,
            eval_set = [(X_val_transformed, y_val)],
            verbose = 50
        )

        
        
        return model   

    def prediction(self, model: Pipeline, 
                   X_val: pd.DataFrame,
                   y_val: pd.Series) -> tuple[np.ndarray,np.ndarray]:
        
        """
        Generate prediction and probabilities.

        Args:
            model (Pipeline): Fitted pipeline
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target

        Returns:
            Tuple of (prediciton, probabilites)
        """

        y_val_pred = model.predict(X_val)
        y_val_prob = model.predict_proba(X_val)[:, 1]
        
        return y_val_pred, y_val_prob

    def eval(self, y_val_pred: pd.Series, 
             y_val: pd.Series) -> dict:
        """
        Evaluate model performance with multiple metrics.

        Args:
            y_val: actuals target
            y_val_prediction: predicted values
        
        """
        acc = accuracy_score(y_val, y_val_pred)
        print("Validation Accuracy: ", acc)

        roc_auc = roc_auc_score(y_val, y_val_pred)
        print("Validation ROC-AUC Score: ", roc_auc)

        precision = precision_score(y_val, y_val_pred)
        print("Validation precision score: ", precision)

        recall = recall_score(y_val, y_val_pred)
        print("Validation Recall score: ", recall)

    def get_test_data(self, dir_):

        """
        Load test data for prediciton.

        """
        
        print("Getting Test Samples....")
        dx  = pd.read_csv(dir_)

        return dx
    

    
    def predict_test(self, X: pd.DataFrame,
                      model: Pipeline,
                     id: pd.Series,
                     output_path: str = 'data/submission.csv') -> None:

        y_pred = model.predict(X)

        dx = pd.DataFrame({
            'PassengerId': id.values,
            'Transported': y_pred
        })

        dx['Transported'] = dx['Transported'].astype(bool)

        dx.to_csv(output_path, index=False)

        return None

    def save_model(self, model: Pipeline, path: str= 'models/xgboost_model.pkl') -> None:
        
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {path}")
        return None
    

    def main(self):

        X_train, y_train, X_val, y_val, X_test, y_test = \
            self.train_test_val_split_dx(self.train_dir)

        X_ = self.get(X_train)
        X_val = self.get(X_val)


        model = self.preprocess_pipeline(X_, X_val, y_train, y_val)

        y_pred, y_prob =  self.prediction(model, X_val, y_val)

        self.eval(y_pred, y_val)

        X = self.get_test_data(self.test_dir)

        X, id = self.get(X, is_train=False)


        self.predict_test(X,model, id)
        self.save_model(model)

        return None
    
    
    

if __name__ == "__main__":

    func_ = ModelPipeline()
    run = func_.main()