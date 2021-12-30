<div align="center">
  
[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]

</div>

# <div align="center">Bank Customer Churn Prediction</div>

<p style='text-align: justify;'> Despite the steady transformation over the decades, many banks today with a sizeable customer base hoping to gain a competitive edge.<br>
While retaining existing customers and thereby increasing their lifetime value is something everyone acknowledges as being important, there is little the banks can do about customer churn when they don’t see it coming in the first place.<br><br>
This is where predicting churn at the right time becomes important, especially when clear customer feedback is absent.Early and accurate churn prediction empowers CRM and customer experience teams to be creative and proactive in their engagement with the customer.</p><br>

<div align="center">
  <img src = "https://github.com/Pradnya1208/Bank-customers-churn-prediction/blob/main/output/gify.gif?raw=true" width="75%">
</div>

## Objectives:
In this project our goal is to predict the probability of a customer is likely to churn using machine learning techniques.
## Dataset:
[Predicting Churn for Bank Customers](https://www.kaggle.com/adammaus/predicting-churn-for-bank-customers)

## Implementation:

**Libraries:** `sklearn` `Matplotlib` `pandas` `seaborn` `NumPy` `Scipy` 


## Few glimpses of EDA:

### 1. Churn Distribution
![Churn](https://github.com/Pradnya1208/Bank-customers-churn-prediction/blob/main/output/churn%20distribution.PNG?raw=true)
 From the above chart, we can say that our traget variable is imbalanced.

### 2. Distribution of the Categorical Variables:
#### a. Grography distribution in customer attrition
![geography](https://github.com/Pradnya1208/Bank-customers-churn-prediction/blob/main/output/geography.PNG?raw=true)

#### b. Gender distribution in customer attrition
![gender](https://github.com/Pradnya1208/Bank-customers-churn-prediction/blob/main/output/gender.PNG?raw=true)

#### c. Customer attrition w.r.t. products
![products](https://github.com/Pradnya1208/Bank-customers-churn-prediction/blob/main/output/products.PNG?raw=true)

#### d. Customer attrition w.r.t. credit card
![credit card](https://github.com/Pradnya1208/Bank-customers-churn-prediction/blob/main/output/creditcard.PNG?raw=true)

#### e. Customer attrition w.r.t. active status of a member
![active](https://github.com/Pradnya1208/Bank-customers-churn-prediction/blob/main/output/iactive.PNG?raw=true)

### 3. Distribution of the continuous Variables:
#### a. Credit Score
![creditscore](https://github.com/Pradnya1208/Bank-customers-churn-prediction/blob/main/output/creditscore%20distribution.PNG?raw=true)

#### b. Age distribution
![age](https://github.com/Pradnya1208/Bank-customers-churn-prediction/blob/main/output/agedistribution.PNG?raw=true)

#### c. Tenure distribution
![tenure](https://github.com/Pradnya1208/Bank-customers-churn-prediction/blob/main/output/tenureditribution.PNG?raw=true)

#### d. Balance distribution
![balance](https://github.com/Pradnya1208/Bank-customers-churn-prediction/blob/main/output/balance%20distribution.PNG?raw=true)

## Model Training and Evaluation:

### Feature Importances
We need to know which the important features are. In order to find that out, we trained the model using the Random Forest classifier.
![Feature importance](https://github.com/Pradnya1208/Bank-customers-churn-prediction/blob/main/output/feature%20importances.PNG?raw=true)
<br>
The graph above shows the features with the highest importance value to the lowest importance value.

### Model Selection
Since we are modeling a critic problem for that we need model with high performance possible. Here, we will try a couple of different machine learning algorithms in order to get an idea about which machine learning algorithm performs better. Also, we will perform a accuracy comparison amoung them. As our problem is a classification problem, the algorithms that we are going to choose are as follows:

- K-Nearest Neighbor (KNN)
- Logistic Regression (LR)
- AdaBoost
- Gradient Boosting (GB)
- RandomForest (RF)

### Base Model results:

<img src = "https://github.com/Pradnya1208/Bank-customers-churn-prediction/blob/main/output/rocauc.PNG?raw=true">
<div align="center">
<img src = "https://github.com/Pradnya1208/Bank-customers-churn-prediction/blob/main/output/baselinemodel.PNG?raw=true" width="60%">
</div>




## Optimizations

### 1.Results after Hyper Parameter Tuning:
#### Adaboost:
```
parameters_list = {"algorithm" : ["SAMME","SAMME.R"],
                  "n_estimators" :[10,50,100,200,400]}
GSA = RandomizedSearchCV(AdaBoostClassifier(), param_distributions=parameters_list, n_iter=10, scoring = "roc_auc")
GSA.fit(X_train, y_train)

```
```
GSA.best_params_, GSA.best_score_
({'n_estimators': 200, 'algorithm': 'SAMME'}, 0.8432902741161931)
```
#### Gradientboost:
```
gb_parameters_list = {'loss' : ['deviance', 'exponential'],
                 'n_estimators': randint(10, 500),
                 'max_depth': randint(1,10)}

GBM = RandomizedSearchCV(GradientBoostingClassifier(), param_distributions=gb_parameters_list, n_iter=10, scoring="roc_auc")
GBM.fit(X_train, y_train)
```
```
GBM.best_params_, GBM.best_score_
({'loss': 'exponential', 'max_depth': 3, 'n_estimators': 241},
 0.8576619853133595)
```

### 2. Results after Feature Transformation:
```
'AdaBoostClassifier': 0.8442783055508478
'GradientBoostingClassifier': 0.873749653401012
```

### 3. Voting Classifier:
```
voting_model = VotingClassifier(estimators=[("gb", GBM_fit_transformed), 
                                            ("ADA", GSA_fit_transformed)],
                                voting = 'soft', weights = [2,1])

                                    
votingModel = voting_model.fit(X_train_transform, y_train)
test_labels_voting = votingModel.predict_proba(np.array(X_test_transform))[:,1]
```
```
votingModel.score(X_test_transform, y_test)
0.8732
```
```
roc_auc_score(y_test,test_labels_voting, average = 'macro', sample_weight = None)
0.8744660402064695
```
### Lessons Learned

`Data Imputation`
`Handling Outliers`
`Feature Engineering`
`Classification Models`
`Voting`

### Feedback

If you have any feedback, please reach out at pradnyapatil671@gmail.com


### 🚀 About Me
#### Hi, I'm Pradnya! 👋
I am an AI Enthusiast and  Data science & ML practitioner


[1]: https://github.com/Pradnya1208
[2]: https://www.linkedin.com/in/pradnya-patil-b049161ba/
[3]: https://public.tableau.com/app/profile/pradnya.patil3254#!/
[4]: https://twitter.com/Pradnya1208


[![github](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c292abd3f9cc647a7edc0061193f1523e9c05e1f/icons/git.svg)][1]
[![linkedin](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/9f5c4a255972275ced549ea6e34ef35019166944/icons/iconmonstr-linkedin-5.svg)][2]
[![tableau](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/e257c5d6cf02f13072429935b0828525c601414f/icons/icons8-tableau-software%20(1).svg)][3]
[![twitter](https://raw.githubusercontent.com/Pradnya1208/Telecom-Customer-Churn-prediction/c9f9c5dc4e24eff0143b3056708d24650cbccdde/icons/iconmonstr-twitter-5.svg)][4]


