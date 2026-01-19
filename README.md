# SpaceshipTitanic - ML Classification Project

A machine learning solution for the [Kaggle Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) that predicts which passengers were transported to an alternate dimension during a spaceship collision with a spacetime anomaly.
Kaggle Score: 0.80+ accuracy | 

## 🗒️ Table of Contents
* [Problem Statements](#problem-statement)
* [Dataset Overview](#dataset-overview) 
* [Approach & Methodology](#approach-and-methodology)
* [Results](#results)
* [Installation & Setup](#installation)
* [Author](#author)


## Problem Statement
The spaceship Titanic was an interstellar passenger liner carrying almost 13,000 passengers when it colleded with a spacetime anamoly. Almost half of the passenger were transported to an alternate dimension. The goal is to predict which passenger were transported based on their personal records.

Task: Binary classification to predict the Transported status (True/False) for each passenger.

## Dataset Overview
The dataset contains personal records recovered from the spaceship's damaged computer system:

Features:
    * PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
    
    * HomePlanet - The planet the passenger departed from, typically their planet of permanent residence.
    
    * CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
    
    * Cabin - The cabin number where the passenger is staying. Takes the form deck/num/ side, where side can be either P for Port or S for Starboard.
    
    * Destination - The planet the passenger will be debarking to.
    
    * Age - The age of the passenger.
    
    * VIP - Whether the passenger has paid for special VIP service during the voyage.
    RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
    
    * Name - The first and last names of the passenger.
    
    * Transported - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

## Approach and Methodology
1. Exploratory Data Analysis (EDA)
    * Analyzed feature distribution and missing values
    * Identified pattern in spending behavior vs CryoSleep
    * Examined correlations between features
    * Visulaized class balanced and relationships

2. Feature Engineering
   Created several new features to improve model performance:
    * GroupSize: Number of passengers in the same travel group
    * TotalSpend: Sum of all the spending across amenties
    * Deck, Num, Side: Extracted from Cabin String
    * Log-transformation: for each spending columns 

3. Feature Selection
Used Boruta algorithm with RandomForest to identify statistically relevant features:

4. Preprocessing Pipeline
   Built sklearn pipeline with:
   * Numeric Features: RobustScaler
   * Categorical feature: OneHotEncoder
   * Modular design: Easy to extend and modify

5. Model Traning
   XGBoost classifier with the following configuration:
   Why XGBoost?
   * Handles mixed feature types well
   * Built-in regularization on validation set
   * Fast training with parallel processing

## Results
   Model Performance
   | Metric   |    Test Set   |  
   |:--------:|:-------------------:|
   | Accuracy | 83.05%              |  
   | ROC-AUC  | 83.06%              |  
   | Precision| 84.04%              |  
   | Recall   | 81.86%              |  
   

## Key Insights:
* Model show consistent performance
* No significant overfitting observed
* Early stopping prevented overtraining (stopped at ~333 iterations)
* Balanced precision and recall indicated good generalization

## Installation
Prerequisites
* Python 3.8 or higher
* pip package manager

```bash
python -m venv venv
source venv/bin/activate
pip install -e .

```

## Author
Priyank
• Role: Sr. Analyst @ E2 Consulting Engineers Inc.
• LinkedIn: [LinkedIn-Profile](https://www.linkedin.com/in/priyank-rao)
• GitHub: [MyGitHub](https://github.com/debugjedi)
• Portfolio: [Portfolio](https://priyankrao.co)


Acknowledgments
• Kaggle for hosting the competition
• The Spaceship Titanic dataset creator
• scikit-learn and XGBoost communities
