#----------------------------------------------EXPLORATORY DATA ANALYSIS-----------------------------------

#--1)--READING THE DATA
#--2)-- DATA CLEANSING
#--3)--DATA ANALYSIS
#--4)--FEATURE ENGINEERING
#--5)--FEATURE SELECTION


#---------------------------------------1)-READING THE DATA-------------------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



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
df["Doors"].unique()



#-----------------------------------------------DATA CLEANSING--------------------------------------------

df1=pd.read_csv("python practise data.csv",na_values=['????','??'],index_col=0)
df1.info()

df1.columns
df1["Price"].unique()
df1[i].unique()


for i in df1.columns:
    print(df1[i].unique(),i)

df1["Price"].value_counts()

for i in df1.columns:
    print(df1[i].value_counts())

#--------------------------------- TWO METHODS TO REPLACE VALUES----------------------------------------
df1.loc[df1["Doors"]=='three',"Doors"]=3

#----------------------------------------alternated method---------------------------------------------

df1["Doors"]=df1["Doors"].replace('four',4)
df1["Doors"].replace("five",5,inplace=True)
df1.Doors.unique()
#----------------------------------------------CONVERSION OF DATA TYPE---------------------------------

df1["MetColor"]=df1["MetColor"].astype("object")
df1["Automatic"]=df1["Automatic"].astype("object")
df1["Doors"]=df1["Doors"].astype("int64")
df1.info()
#------------------------------------------------------DATA ANALYSIS-----------------------------------

#---1---------------------------------Segregation of data in numerical and categorical-----------------

#--------------------------------------------------1-Method---------------------------------------------
df1_num=df1.select_dtypes(exclude="object")
df1_num.columns


df1_ob= df1.dtypes[df1.dtypes=="object"]
df1_ob
df1_ob=pd.DataFrame(df1_ob).T
a=df1_ob.columns
a
df1_ob=df1[a]

df1_num.describe()
df1_ob.describe()

#------------------------------------------Alternate Method--------------------------------------------
df1_num=df1.select_dtypes(exclude="object")
df2_ob=df1.drop(df1_num,axis=1)
df1_ob=df1.select_dtypes(include="object")
#------2-------------------------------------Do Visualisation------------------------------------------

sns.set(rc={"figure.figsize":(40,30)})





#-----------------------------------------------temporary drop missing values--------------------------

df1_num.isnull().sum()
df1_ob.isnull().sum()

df1_num1=df1_num.dropna()
df1_ob1=df1_ob.dropna()
df1_num1.info()
df1_ob1.info()

#-----------------------------------------------univariate analysis-----------------------------------

plotnumber=1
for i in df1_num1:
    if plotnumber<=9:
        ax=plt.subplot(3,3,plotnumber)
        sns.distplot(df1_num1[i])
        plt.xlabel(i,fontsize=20)
        plotnumber+=1
plt.show()





plotnumber=1
for i in df1_ob1:
    if plotnumber<=9:
        ax=plt.subplot(3,3,plotnumber)
        sns.countplot(df1_ob1[i])
        plt.xlabel(i,fontsize=20)
        plotnumber+=1
plt.show()

#----------------------------------------------Bivariate Analysis------------------------------------

plotnumber=1
for i in df1_ob1:
    for j in df1_num1:
        if plotnumber<=25:
            ax=plt.subplot(5,5,plotnumber)
            sns.barplot(df1_ob1[i],df1_num1[j])
            plt.xlabel(i,fontsize=20)
            plotnumber+=1
plt.show()


plotnumber=1
for i in df1_num1:
    for j in df1_ob1: 
        if plotnumber<=25:
            ax=plt.subplot(5,5,plotnumber)
            sns.boxplot(y=df1_num[i],x=df1_ob[j])
            plt.xlabel(j)
            plt.ylabel(i)
            plotnumber+=1
plt.show()

plotnumber=1
for i in df1_num1:
    for j in df1_ob1: 
        if plotnumber<=25:
            ax=plt.subplot(5,5,plotnumber)
            sns.stripplot(y=df1_num1[i],x=df1_ob[j])
            plt.xlabel(j)
            plt.ylabel(i)
            plotnumber+=1
plt.show()

sns.heatmap(df1_num.corr(),annot=True)

plt.figure(figsize=(20,25), facecolor='white')

plotnumber=1
for i in [ 'Age', 'KM', 'HP', 'CC', 'Doors', 'Weight']:
    if plotnumber<=9:
        ax=plt.subplot(3,3,plotnumber)
        plt.scatter(y=df1_num1["Price"],x=df1_num1[i])
        plt.xlabel(i)
        plt.ylabel("Price")
        plotnumber+=1
plt.show()

