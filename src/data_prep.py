import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import seaborn as sns

key_sep = "*"
sep_count = 25
sec_sep = "="


class Preparation():
    def __init__(self, dir_train = r"./data/raw/train.csv"):
        self.train_dir = dir_train
        
    

    def get(self):
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        X  = pd.read_csv(self.train_dir)
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

        summary = pd.DataFrame({
                    "dtypes": X.dtypes,
                    "unique": X.nunique(),
                    "non-nulls": X.count(),
                    "null": X.isna().sum(),
                    "min": X.min(numeric_only=True),
                    "max": X.max(numeric_only=True),
                })
        summary["%null"] = (summary["null"]/summary["non-nulls"])*100
        print(summary)




          
        return X
    
    def descriptive(self, dx):
        print("Descriptive Stats....")
        print(sec_sep*sep_count, key_sep*sep_count, sec_sep*sep_count)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        
        
        print(dx.head(20))
        
        num_cols = dx.select_dtypes(include='number').columns
        print("Numeric columns: ", num_cols)

        print("Data Representation: ")
        print(sec_sep*sep_count, key_sep*sep_count, sec_sep*sep_count)
        summary = pd.DataFrame({
            "dtypes": dx.dtypes,
            "unique": dx.nunique(),
            "non-nulls": dx.count(),
            "null": dx.isna().sum(),
            "min": dx.min(numeric_only=True),
            "max": dx.max(numeric_only=True),
        })
        summary["%null"] = (summary["null"]/summary["non-nulls"])*100
        print(summary)
        
        print(key_sep*sep_count,"Descriptive stats",key_sep*sep_count)
        print(dx.describe())

        print(dx.groupby(["Transported"]).size().reset_index(name= "Count"))
        # for col in num_cols:
        #     plt.figure(figsize=(6,4))
        #     plt.hist(dx[col], bins=30, color='skyblue', edgecolor= 'black')
        #     plt.title(f"Histogram of {col}")
        #     plt.xlabel(col)
        #     plt.ylabel('Frequency')
        #     plt.show()
        
        
        return dx
    
    def collienarityCheck(self, dx, num_Cols):
        
        num_data = dx[num_Cols]
        corr = num_data.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0
        )

        plt.title("Correlation Matrix (numeric features)")
        plt.show()



    def featureimportance(self, dx):
        X = dx.drop(columns=['Transported']).copy()
        y = dx['Transported'].astype(int).copy()

        for col in X.columns:
            if X[col].dtype == 'object' or X[col].name == 'category':
                # Fill missing categorical with 'Missing' and encode as numbers
                X[col] = X[col].fillna('Missing').factorize()[0]
            else:
                # Fill missing numeric with median
                X[col] = X[col].fillna(X[col].median())

        x_np = X.values
        y_np = y.values

        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=5
        )

        boruta = BorutaPy(
            estimator=rf,
            n_estimators="auto",
            verbose=2,
            random_state=42
        )
        boruta.fit(x_np, y_np)

        selected_feature = X.columns[boruta.support_]
        tentative_feature = X.columns[boruta.support_weak_]
        feature_ranks = pd.DataFrame({
            "feature": X.columns,
            "rank": boruta.ranking_,
            "selected": boruta.support_
        }).sort_values("rank")

        return selected_feature, tentative_feature, feature_ranks

    def main(self):
        dx = self.get()
        dx = self.descriptive(dx)
        selected_feature, tentative_feature, feature_rank = self.featureimportance(dx)

        num_cols = dx.select_dtypes(include=["int64", "float64"]).columns
        self.collienarityCheck(dx, num_cols)
        print("Selected Feature...")
        print(sec_sep*sep_count, key_sep*sep_count, sec_sep*sep_count)
        print(selected_feature)
        print(sec_sep*sep_count, key_sep*sep_count, sec_sep*sep_count)
        print("feature_rank..")
        print(feature_rank)
        return None
    


if __name__ == "__main__":
    prep = Preparation()
    run  = prep.main()
