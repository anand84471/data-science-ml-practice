#----------------------------------------------LINEAR REGRESSION MODEL----------------------------------------

#--1)--READING THE DATA
#--2)-- DATA CLEANSING
#--3)--DATA ANALYSIS
#--4)--FEATURE ENGINEERING
#--5)--FEATURE SELECTION


import os

import matplotlib.pyplot as plt
import numpy as np
#---------------------------------------1)-READING THE DATA-------------------------------------------------
import pandas as pd
import seaborn as sns

os.getcwd()


#--------------------------------------------------SET OPTIONS--------------------------------------------
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
pd.set_option("float_format",lambda x :"%.2f" %x)

#--------------------------------------------------IMPORT THE DATA----------------------------------------
df=pd.read_csv("python practise data.csv")
df.isnull().sum()
df.info()
df["KM"].unique()
df["HP"].unique()
a=df["KM"].value_counts()
a



data_cancel=df.loc[df["Age"].isnull()==True]




#-----------------------------------------------DATA CLEANSING--------------------------------------------

df1=pd.read_csv("python practise data.csv",na_values=['????','??'],index_col=0)
df1.info()



#--------------------------------- TWO METHODS TO REPLACE VALUES----------------------------------------
df1.loc[df1["Doors"]=='three',"Doors"]=3

#----------------------------------------alternated method---------------------------------------------

df1["Doors"]=df1["Doors"].replace('four',4)
df1["Doors"].replace("five",5,inplace=True)

#----------------------------------------------CONVERSION OF DATA TYPE---------------------------------

df1["MetColor"]=df1["MetColor"].astype("object")
df1["Automatic"]=df1["Automatic"].astype("object")
df1["Doors"]=df1["Doors"].astype("int64")

#---1---------------------------------Segregation of data in numerical and categorical-----------------

#--------------------------------------------------Method---------------------------------------------

df1_num=df1.select_dtypes(exclude="object")
df1_ob=df1.drop(df1_num,axis=1)





#-----------------------------------------------temporary drop missing values--------------------------

df1_num.isnull().sum()
df1_ob.isnull().sum()




#------------------------------------Feature Engineering----------------------------------------------

# 1) Missing values
# 2) outliers
# 3) Feature generation
# 4) convert categorical attributes into numerical according to classification of data
# 5) standard scaling




.............--------------------Fill Missing Values with Variance --------------------------------------------------------------

#--------------here, we will be seen, the variance of numerical feature with categorical--------------

#--------------------fill nominal categorical features with their mode-------------------------------

df1_ob["FuelType"]=df1_ob["FuelType"].fillna(df1_ob["FuelType"].mode()[0])

for i in df1_ob:
    df1_ob[i]=df1_ob[i].fillna(df1_ob[i].mode()[0])      


#-----------------------to extract out merely those columns which are having missing values----------

      


df3_num=df1_num.copy()

df3_num["Age"]=df3_num.groupby(df1_ob["MetColor"])["Age"].apply(lambda x : x.fillna(x.mean()))

for i in ["KM","HP"]:
    df3_num[i]=df3_num.groupby(df1_ob["FuelType"])[i].apply(lambda x:x.fillna(x.mean()))

       


##-------------------------------------------------OUTLIERS------------------------------------------


#--1)- Z-score method



#------------------------------------------Z-SCORE OR 3-SIGMA-METHOD---------------------------------------

min_3sigma,max_3sigma=(df3_num["Age"].mean()-(3*(df3_num["Age"].std()))),(df3_num["Age"].mean()+(3*(df3_num["Age"].std()))) 
min_3sigma,max_3sigma

df_out_3sigma=df3_num.loc[(df3_num["Price"]>min_3sigma) & (df3_num["Price"]<max_3sigma)]

#--------------------------------code for whole data in one go----------------------------------------

df1_final=df3_num.copy()

def sigma_outlier():
    min_sigma=[]
    max_sigma=[]
    for i in df3_num:
        min_3sigma,max_3sigma=df3_num[i].mean()-3*df3_num[i].std(),df3_num[i].mean()+3*df3_num[i].std()
        min_sigma.append(min_3sigma)
        max_sigma.append(max_3sigma)
    return (min_sigma,max_sigma)
        
len(sigma3[0])
sigma3=sigma_outlier()
sigma3[0][0]
sigma3[1][0]

j=0
for i in df3_num:
  if j<=len(sigma3[0])-1:
      df1_final.loc[df1_final[i]<=sigma3[0][j],i]=sigma3[0][j]
      df1_final.loc[df1_final[i]>=sigma3[1][j],i]=sigma3[1][j]
      j+=1





#--------------------------------------CONVERT ALL CATEGORICAL INTO NUMERICAL-------------------------
        
#----------------- one hot encoding----(merely for nominal feature)-------------------------------------------

df1_final=pd.concat([df1_final,df1_ob],axis=1)

df1_final_OHC=pd.get_dummies(df1_final,drop_first=True)   


#------------------------------------------------3-LASSO REGRESSION-----------------------------------

X=df1_final_OHC.drop("Price",axis=1)
y=df1_final_OHC["Price"]

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LassoCV

reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)

reg.coef_
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

imp_coef = coef.sort_values()
imp_coef


ls=SelectFromModel(Lasso(alpha=88625.861600,random_state=0))
ls.fit(X,y)
ls.get_support()

selected_features=X.columns[(ls.get_support())]

selected_features

#-------------------------------------ANOTHER METHOD FOR FEATURE SELECTION----------------------------------------------------

import statsmodels.api as sm

x=df1_final_OHC.drop(["Price"],axis=1)
y=df1_final_OHC["Price"]

#-------------------------------------ADD CONSTANT IN X DATAFRAME--------------------------------

x=sm.add_constant(x)

#----------------------FIT A MODEL WITH INTERCEPT------------------------------------------------

ols_model=sm.OLS(y,x).fit()
ols_model.summary()

cols = list(x.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = x[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)

#----------------------------------------------CROSS-CHECK MULTICOLLINEARITY WITH VIF-------------------------------
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif=pd.DataFrame()

vif["VIF"]=[variance_inflation_factor(x.values,i) for i in range(x.shape[1])]

vif["column"]=x.columns

#------------------------------------------------------DROP FUEL_TYPE_Diesel COLUMN-------------------------------------

best_model1=df1_final_OHC.drop(["FuelType_Diesel"],axis=1)

#--------------------------------------------------------------APPLY LINEAR REGRESSION MODEL---------------------------


..........................................................SCALING.........................................................
X=best_model1.drop(["Price"],axis=1)
Y=best_model1["Price"]



from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

best_model=scaler.fit_transform(X)

best_model=pd.DataFrame(best_model,columns=X.columns)



from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(best_model,Y,test_size=0.25,random_state=0)

from sklearn.linear_model import LinearRegression

lr=LinearRegression()

lin_model=lr.fit(X_train,Y_train)

#------------------------------------------R^2 score -----------------------------------------------

lr.score(X_train,Y_train)
lr.score(X_test,Y_test)

#----------------------------------------ADJUSTED R^2------------------------------------------
def adj_r2(x,y):
    r2 = lr.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2

adj_r2(X_train,Y_train)

adj_r2(X_test,Y_test)
#..........................................................................................................
from sklearn.metrics import mean_absolute_error, mean_squared_error

Y_pred=lin_model.predict(X_test)

print(mean_squared_error(Y_pred,Y_test))
print(np.sqrt(mean_squared_error(Y_pred,Y_test)))

sns.regplot(Y_test,Y_pred)

sns.distplot(Y_test-Y_pred)



#-----------------------------CHECKING OVERFITTING OR REGULARIZATION--------------------------------------

from sklearn.linear_model import (ElasticNet, ElasticNetCV, Lasso, LassoCV,
                                  Ridge, RidgeCV)

#------------------------------------------1) BY RIDGE REGRESSION--------------------------------------------------------------

#-----RidgeCV will return best alpha and coefficients after performing 10 cross validations------------------

alphas = np.random.uniform(low=0, high=10000, size=(500,))
ridgecv = RidgeCV(alphas = alphas,cv=10,normalize = True)
ridgecv.fit(X_train, Y_train)

#-----------------------------------BEST ALPHA VALUE( HYPERPARAMETER OPTIMISATION)------------------------------------------------
ridgecv.alpha_

#----------------------------------------------RIDGE MODEL-----------------------------------------------

ridge_model=Ridge(alpha=ridgecv.alpha_)
rid_model=ridge_model.fit(X_train,Y_train)

rid_model.score(X_train,Y_train)
rid_model.score(X_test,Y_test)



#----------------------------------------2) BY LASSO MODEL-----------------------------------------------

alphas = np.random.uniform(low=0, high=10000, size=(500,))
lassocv = LassoCV(alphas = None,cv=10,normalize = True,max_iter=10000)
lassocv.fit(X_train, Y_train)

#-------------------------------------BEST ALPHA VALUE(HYPERPARAMETER OPTIMISATION)----------------------

lassocv.alpha_

#---------------------------------------------------LASSO MODEL------------------------------------------

lasso_model=Lasso(alpha=lassocv.alpha_)
las_model=lasso_model.fit(X_train,Y_train)

las_model.score(X_train,Y_train)
las_model.score(X_test,Y_test)

#-------------------------------------3) BY ELASTIC NET--------------------------------------------------



elasticCV = ElasticNetCV(alphas=None , cv =10,max_iter=10000,normalize=True)

elasticCV.fit(X_train, Y_train)

#----------------------------------BEST ALPHA VALUE( HYPERPARAMETER OPTIMISATION)-------------------

elasticCV.alpha_

#- l1_ration gives how close the model is to L1 regularization, below value indicates we are giving equal
#- preference to L1 and L2

elasticCV.l1_ratio

#-----------------------------------------------ELASTIC NET MODEL----------------------------------------

elastic_model=ElasticNet(alpha = elasticCV.alpha_,l1_ratio=0.5)
elas_model=elastic_model.fit(X_train, Y_train)

elas_model.score(X_train,Y_train)
elas_model.score(X_test,Y_test)


#------------------------------------------MODEL BUILDING------------------------------------------------

#------------------------------------------1)LIN-MODEL---------------------------------------------------

from sklearn.metrics import mean_absolute_error, mean_squared_error

Y_pred=lin_model.predict(X_test)

print(mean_squared_error(Y_pred,Y_test))
print(np.sqrt(mean_squared_error(Y_pred,Y_test)))

sns.regplot(Y_test,Y_pred)

sns.distplot(Y_test-Y_pred)

#-------------------------------------------2)RID-MODEL---------------------------------------------------
Y_pred=rid_model.predict(X_test)

print(mean_squared_error(Y_pred,Y_test))
print(np.sqrt(mean_squared_error(Y_pred,Y_test)))

sns.regplot(Y_test,Y_pred)

sns.distplot(Y_test-Y_pred)

#----------------------------------------------3) LASSO- MODEL-------------------------------------------
Y_pred=las_model.predict(X_test)

print(mean_squared_error(Y_pred,Y_test))
print(np.sqrt(mean_squared_error(Y_pred,Y_test)))

sns.regplot(Y_test,Y_pred-Y_test)

sns.distplot(Y_test-Y_pred)

#----------------------------------------------4) ELASTIC NET---------------------------------------------
Y_pred=elas_model.predict(X_test)

print(mean_squared_error(Y_pred,Y_test))
print(np.sqrt(mean_squared_error(Y_pred,Y_test)))

sns.regplot(Y_test,Y_pred)

sns.distplot(Y_test-Y_pred)




