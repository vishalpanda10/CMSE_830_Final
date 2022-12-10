import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
from os import path
import streamlit as st
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import sklearn
from PIL import Image
import warnings
from sklearn.neighbors import KNeighborsClassifier  

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler,OrdinalEncoder,MinMaxScaler,MaxAbsScaler,RobustScaler,normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,f1_score,precision_score,recall_score,roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV


st.sidebar.write("# **Heart Disease Prediction Models**")



warnings.filterwarnings('ignore')

sns.set_style("whitegrid")

colors6 = sns.color_palette(['#1337f5', '#E80000', '#0f1e41', '#fd523e', '#404e5c', '#c9bbaa'], 6)
colors2 = sns.color_palette(['#1337f5', '#E80000'], 2)
colors1 = sns.color_palette(['#1337f5'], 1)

df = pd.read_csv("heart.csv")

st.set_option('deprecation.showPyplotGlobalUse', False)

def show_relation(col, according_to, type_='dis'):
  plt.figure(figsize=(15,7));

  if type_=='dis':
    sns.displot(data=df, x=col, hue=according_to, kind='kde', palette=colors2);
  elif type_=='count':
    if according_to != None:
      perc = df.groupby(col)[according_to].value_counts(normalize=True).reset_index(name='Percentage')
      sns.barplot(data=perc, x=col,y='Percentage', hue=according_to, palette=colors6, order=df[col].value_counts().index);
    else:
      sns.countplot(data=df, x=col, hue=according_to, palette=colors1, order=df[col].value_counts().index);

  if according_to==None:
    plt.title(f'{col}');
  else: 
    plt.title(f'{col} according to {according_to}');
    
    
def generate_colors(num):
    colors = []
    lst = list('ABCDEF0123456789')

    for i in range(num):
        colors.append('#'+''.join(np.random.choice(lst, 6)))
        
    return colors


def Introduction():
    st.write("# **Indicators of Heart Disease**")
    st.write("## **Introduction**")
    st.write("According to the world health organization, Cardiovascular diseases (CVDs) are the leading cause of death globally. In 2019 alone, around 17.9 million people died from CVDs. Of these deaths, 85% of them were due to heart diseases. There are many factors that play a role in increasing the risk of heart disease. Identifying these factors and their impact is paramount in the field of healthcare. Identifying patients who are at greater risk enables medical professionals to respond quickly and efficiently, saving more lives.")
     #st.selectbox()

    st.write("In this project we will delve deep into the causes of heart disease and its relations with other health indicators, drawing insights and exploring the data in order to get a better picture of the leading causes.")
    from PIL import Image
    #image=Image.open("C:/Users/visha/Downloads/heart.jpg")
    st.image("https://www.kaggleusercontent.com/kf/107235006/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..4QLPzVXFb5D2NBoGfSVkUQ.-5jgWDnaJZe58QcPIucHKlK_9X_l-cHqjAXKxLTZByhMWa45VO_r3D-Lq8XqId_XNhophAnYhkx8YM0zu-6I2ic5MM7QPNY0pXC8fGiQNPH3Oz1yvWJ8kfjfNJhdE07SQFkJmwK9K3bOkLgez_6lwZo--m1PXppvAhrwCjO_FjEUkIWaN9NISnrLFSSOxevPvDDbiEcWq7slOVxw6vQR73fqZElv9ji-JxhAe4hqT2FSgutm39WIyPytZViP-SOKZKU6uabxLihsnVtH2pJaFxmJL0IBSwSlOVlgPIfkKY6q0tPiVxkmAFHy6_tHeLJSKj6uKQbgC6pj7n-c1SUQflJAk7kVDg_NlNw4oHG-MW-sbHy9aNtbShYhMb8GhGV6y_newdS52Ou5bjAEbdFHz6xyWVR81KsW1pn-xZviOOUoEQXuoFVdkAtMiS90EG5kphGUFJDwB6V8vtayIaqBeWibeBs9tWcn6QUU6f2dIkjuvpyrnw4Z3ooLAtN78VWxCswl4VNtXWgyo2UvmLzgZHaDQcLjbOd_mgDPqFSTnNQdBcqTNVZqYDNgt4gQypse-SAZT-cVbxidSDAYOr9QncfyCcXnchIjrWNJhlB-X2T5y9UkmrxBn4uuoFXXZW-U26y5RVHg23SyWYTgSiwj-uam-dilLW5j_GdzTPDcNwnAwUfTYBVM-bFjJ4gqstVj.7M0xFjCzQ-GDZaHBwPCQxg/__results___files/__results___31_0.png")

    if st.button("Glimpse of Heart Data"):
        st.write(df.head())

    fig=plt.figure(figsize=(15,7));
    plt.title('HeartDisease Count');
    sns.countplot(data=df, x='HeartDisease', palette=colors2, order=df['HeartDisease'].value_counts().index);
    st.pyplot(fig)



     ####




    st.write("**About the Dataset:**")
    st.write("""
     The dataset contains 320K rows and 18 columns. It is a cleaned, smaller version of the 2020 annual CDC (Centers for Disease Control and Prevention) survey data of 400k adults. For each patient (row), it contains the health status of that individual. The data was collected in the form of surveys conducted over the phone. Each year, the CDC calls around 400K U.S residents and asks them about their health status, with the vast majority of questions being yes or no questions. Below is a description of the features collected for each patient:
     
     
     """)

    if st.button("Glimpse of Data Columns"):
        from PIL import Image
        #im=Image.open("https://drive.google.com/file/d/1IPpgndT1pAHyV9AhD8nw49DmDRN9ekqC/view?usp=share_link")
        st.image("https://drive.google.com/file/d/1IPpgndT1pAHyV9AhD8nw49DmDRN9ekqC/view?usp=share_link")


    st.write("**Initial Data Assessment of Numerical Values:**")
    categorical=list(set(df.columns)-set(['BMI','SleepTime','PhysicalHealth','MentalHealth']))
    numerical=list(['BMI','SleepTime','PhysicalHealth','MentalHealth'])

    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches(15, 9)
    fig.tight_layout(pad=5)
    k = 0
    numerical_variables=numerical
    for num_var in numerical_variables:
         k = int(numerical_variables.index(num_var)/2)
         sns.distplot(df[num_var], ax=axes[k, numerical_variables.index(num_var)%2])

    plt.plot()
    plt.show()
    st.pyplot(fig)
    st.write("""
    - BMI is skewed.  
    - Physical Health & Mental Health are severely imbalanced due to number of zeroes.
    - SleepTime is normally distributed.
                """)



         #################
    obj_cols = df.select_dtypes(include='object').columns[1:]
    num_cols = df.select_dtypes(exclude='object').columns


    fig=plt.figure(figsize=(20, 40))
    for i in range(len(obj_cols)):
      plt.subplot(7, 2, i+1)

      if(df[obj_cols[i]].nunique() < 3):
        ax = sns.countplot(data=df, x=obj_cols[i], palette=colors2, order=df[obj_cols[i]].value_counts().index[:6])
      else:
        ax = sns.countplot(data=df, x=obj_cols[i], palette=colors6, order=df[obj_cols[i]].value_counts().index[:6])

  
      plt.title(f'{obj_cols[i]}', fontsize=15, fontweight='bold', color='brown')
      plt.subplots_adjust(hspace=0.5)

      for p in ax.patches:
        height = p.get_height() 
        width = p.get_width()
        percent = height/len(df)

        ax.text(x=p.get_x()+width/2, y=height+2, s=format(percent, ".2%"), fontsize=12, ha='center', weight='bold')
    st.pyplot(fig)

    st.write(""""
    - Most of people in our data are white and have no diabetic
    - A litle of them who have asthma, kidney disease and skin cancer.    
    """)

  
    

    obj_cols = df.select_dtypes(include='object').columns[1:]
    num_cols = df.select_dtypes(exclude='object').columns

    
    features=st.selectbox("Feature Selection",obj_cols)


    


    obj_cols = df.select_dtypes(include='object').columns[1:]
    num_cols = df.select_dtypes(exclude='object').columns
    features=st.selectbox("Feature Selection",num_cols)
    if features=='BMI':
        st.write("Is the BMI of heart disease patients different?")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        fig=show_relation(num_cols[0], 'HeartDisease')
        st.pyplot(fig)

        fig=plt.figure(figsize=(16, 6), dpi=80)

        sns.boxplot(data=df, x='BMI', y='HeartDisease', saturation=0.4,width=0.15, boxprops={'zorder': 2},showfliers = False, whis=0,  palette=colors2);
        sns.violinplot(data=df, x='BMI', y='HeartDisease',inner='quartile', palette=colors2);
        st.pyplot(fig)

        st.write("""
        - the both distributions are normal distriubtions and in the same range which is from 10 t0 90 approximately.
        - The BMI distribution of individuals who suffer from heart disease is slightly shifted towards higher values in comparison to the distribution of those who don't. 
        """)

        st.write("Does BMI differ across diseases?")

        fig, ax = plt.subplots(figsize = (14,6))
        sns.kdeplot(df[df["HeartDisease"]=='Yes']["BMI"], alpha=1,shade = False, color=colors6[0], label="HeartDisease", ax = ax)
        sns.kdeplot(df[df["KidneyDisease"]=='Yes']["BMI"], alpha=1,shade = False, color=colors6[1], label="KidneyDisease", ax = ax)
        sns.kdeplot(df[df["SkinCancer"]=='Yes']["BMI"], alpha=1,shade = False, color=colors6[2], label="SkinCancer", ax = ax)
        sns.kdeplot(df[df["Asthma"]=='Yes']["BMI"], alpha=1,shade = False, color=colors6[3], label="Asthma", ax = ax)
        sns.kdeplot(df[df["Stroke"]=='Yes']["BMI"], alpha=1,shade = False, color=colors6[4], label="Stroke", ax = ax)
        sns.kdeplot(df[df["Diabetic"]=='Yes']["BMI"], alpha=1,shade = False, color=colors6[5], label="Diabetic", ax = ax)


        ax.set_xlabel("BMI")
        ax.set_ylabel("Frequency")
        ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
        plt.show()
        st.pyplot(fig)

    if features=='PhysicalHealth':
        st.pyplot(show_relation(num_cols[1], 'HeartDisease'))


        fig, ax = plt.subplots(figsize = (14,6))
        sns.kdeplot(df[df["HeartDisease"]=='Yes']["PhysicalHealth"], alpha=1,shade = False, color=colors6[0], label="HeartDisease", ax = ax)
        sns.kdeplot(df[df["KidneyDisease"]=='Yes']["PhysicalHealth"], alpha=1,shade = False, color=colors6[1], label="KidneyDisease", ax = ax)
        sns.kdeplot(df[df["SkinCancer"]=='Yes']["PhysicalHealth"], alpha=1,shade = False, color=colors6[2], label="SkinCancer", ax = ax)
        sns.kdeplot(df[df["Asthma"]=='Yes']["PhysicalHealth"], alpha=1,shade = False, color=colors6[3], label="Asthma", ax = ax)
        sns.kdeplot(df[df["Stroke"]=='Yes']["PhysicalHealth"], alpha=1,shade = False, color=colors6[4], label="Stroke", ax = ax)
        sns.kdeplot(df[df["Diabetic"]=='Yes']["PhysicalHealth"], alpha=1,shade = False, color=colors6[5], label="Diabetic", ax = ax)


        ax.set_xlabel("PhysicalHealth")
        ax.set_ylabel("Frequency")
        ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
        plt.show()
        st.pyplot(fig)


    if features=='MentalHealth':
        st.write("Are heart disease patients more mentally unwell?")




        fig=show_relation(num_cols[2], 'HeartDisease')
        st.pyplot(fig)



        



    if features=='SleepTime':

        st.write("Distribution of sleep time among heart disease patients")

        st.pyplot(show_relation(num_cols[3], 'HeartDisease'))

        relative = df.groupby('HeartDisease').SleepTime.value_counts(normalize=True).reset_index(name='Percentage')

        fig=plt.figure(figsize=(16, 6), dpi=80)
        ax = sns.barplot(data=relative, x='SleepTime', y='Percentage', hue='HeartDisease', palette=colors2);

        ax.set_title("Percentage of Sleep Times by Heart Disease");

        st.pyplot(fig)

        st.write("""
        - Abnormal sleeep duration is more prevalent in heart disease patients.
        - Higher percentages of sleep less than 6 hours or more than 9 hours have heart disease.
        """)




    st.write("## **Summary**")
    st.write("""
    - Around 9 among 100 individuals suffer from heart disease
    - The BMI of heart disease patients is slightly higher than that of healthy individuals.
    - The older the individual, the more susceptible they are to heart disease.
    - A lot more people who suffer from heart disease say they have poor or fair health compared to those who don't
    - 79% of healthy individuals have been physically active in the past 30 days, compared to 64% in heart disease patients.
    - Abnormal sleeep duration is more prevalent in heart disease patients
    - people who smoke suffer from heart disease
    - Those who have suffered from kidney disease & Asthma are at a sginificantly higher risk of heart disease.
    - Mental health, sleep duration, and physical health are similar among people who suffer from different diseases.
    - Having a stroke is highly correlacted with heart disease.
    




    """)

def Page4():
    heart=pd.read_csv("heart_2020_ml.csv")
    
    st.write("## **Data Pre-Processing & Feature Engineering**")
    features=st.multiselect("Number of Features",['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke',
       'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime',
       'Asthma', 'KidneyDisease', 'SkinCancer'],['BMI','GenHealth','SleepTime','PhysicalHealth', 'MentalHealth','Smoking','AlcoholDrinking','DiffWalking'])
    X=np.array(heart[features])
    y=np.array(heart[['HeartDisease']])
    n=st.slider("Test Size",0.0,1.0,0.3)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = n, random_state = 0)
    
    

    scaling=st.selectbox("Scaling",('No Scaling','Absolute Maximum Scaling','Min-Max Scaling','Normalization','Standard Scaling','Robust Scaling'))
    if scaling=='No Scaling':
        x_train = x_train
        x_test =x_test

    
    
    if scaling=='Absolute Maximum Scaling':
        asc = MaxAbsScaler()
        x_train = asc.fit_transform(x_train)
        x_test = asc.transform(x_test)

    if scaling=='Min-Max Scaling':
        sc = MinMaxScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)
    if scaling=='Normalization':
        x_train = normalize(x_train)
        x_test = normalize(x_test)

    if scaling=='Standard Scaling':
        ssc = StandardScaler()
        x_train = ssc.fit_transform(x_train)
        x_test = ssc.transform(x_test)
    if scaling=='Robust Scaling':
        rsc = RobustScaler()
        x_train = rsc.fit_transform(x_train)
        x_test = rsc.transform(x_test)







    st.write("# **Machine Learning Models to Classify Whether a Person is having Heart Disease or not**")
    tab1,tab2,tab3,tab4=st.tabs([' Logistic Regression','Random Forest','K-Nearest Neighbor','Support Vector Machine'])

    

    with tab1:

        with st.expander("Logistic Regression"):
                st.write(              """
                    Logistic Regression was used in the biological sciences in early twentieth century. It was then used in many social science applications. Logistic Regression is used when the dependent variable(target) is categorical.

For example,

To predict whether an email is spam (1) or (0)
Whether the tumor is malignant (1) or not (0)
Consider a scenario where we need to classify whether an email is spam or not. If we use linear regression for this problem, there is a need for setting up a threshold based on which classification can be done. Say if the actual class is malignant, predicted continuous value 0.4 and the threshold value is 0.5, the data point will be classified as not malignant which can lead to serious consequence in real time.
                 
                    """
                    
                    )
                st.image("https://miro.medium.com/max/1400/1*RqXFpiNGwdiKBWyLJc_E7g.webp")

        

        
        
        





                    
            
            

        col1,col2,col3= st.columns(3)
        with col1:
            solvers=st.selectbox("Select Solver",('newton-cg', 'lbfgs', 'liblinear'))
        with col2:
            c_values=st.slider("Alpha",0.001,1.0,1.0)
        with col3:
            pen=st.selectbox("Penalty",('none','l1','l2','elasticnet'))
            
     

        lr=LogisticRegression(penalty=pen,C=c_values,solver=solvers)
        lr.fit(x_train, y_train)
        pred= lr.predict(x_test)
        lr_probs = lr.predict_proba(x_test)[::,1]
        fpr, tpr, _ =roc_curve(y_test,lr_probs)
        
        accuracyscore=accuracy_score(y_test,pred)
        confusion=confusion_matrix(y_test,pred)
        f1score=f1_score(y_test,pred)
        precision=precision_score(y_test,pred)
        recall=recall_score(y_test,pred)
        st.write("#### Accuracy:",round(accuracyscore,2),   "\t Precision:",round(precision,2),"\t Recall:",round(recall,2),"\t F1 Score:",round(f1score,2))
        report = classification_report(y_test,pred, output_dict=True)

        
        
        st.write("Gridsearch is used to find the best hyperparameters and a stratified K-fold cross validation is used to evaluate the model.")
        st.write("Use to get the best Optimized Hyperparameters :")
        if st.button("GridSearch LR"):
            solvers = ['newton-cg', 'lbfgs', 'liblinear']
            penalty = ['l2']
            c_values = [1.0, 0.1, 0.01]
            # define grid search
            grid = dict(solver=solvers,penalty=penalty,C=c_values)
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            grid_search = GridSearchCV(estimator=lr, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
            grid_result = grid_search.fit(X,y)
            # summarize results
            st.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))        

        fig=px.imshow(confusion,text_auto=True,labels=dict(x="Actual Label", y="Predicted Label"),x=['Heart Disease','No Heart Disease'],y=['Heart Disease','No Heart Disease'])
        fig.update_layout(title_text='Confusion Matrix', title_x=0.5)
        st.plotly_chart(fig)
        

        fig=px.line(fpr,tpr,labels=dict(x="False Positive Rate", y="True positive Rate"))

        fig.update_layout(title_text='ROC Curve',title_x=0.5)
        st.plotly_chart(fig)

               
        
   
          
        

    with tab2:
        with st.expander("Random Forest"):
            st.write("""
            Random forest, like its name implies, consists of a large number of individual decision trees that operate as an ensemble. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model's prediction (see figure below).
            
            
            """)
            st.image("https://miro.medium.com/max/1052/1*VHDtVaDPNepRglIAv72BFg.webp")

        
        col1,col2,col3= st.columns(3)
        with col1:
            estimators=st.slider("Estimators",0,1000,100)
        with col2:
           criter=st.selectbox("Split Criterion",('gini', 'entropy', 'log_loss'))
        
        with col3:
            depth=st.slider("Maximum Depth",1,10,5)

        rf=RandomForestClassifier(n_estimators=estimators,criterion=criter, max_depth=depth)
        rf.fit(x_train, y_train)
        pred= rf.predict(x_test)
        fpr, tpr, _ =roc_curve(y_test,pred)
        
        accuracyscore=accuracy_score(y_test,pred)
        confusion=confusion_matrix(y_test,pred)
        f1score=f1_score(y_test,pred)
        precision=precision_score(y_test,pred)
        recall=recall_score(y_test,pred)
        st.write("#### Accuracy:",round(accuracyscore,2),   "\tPrecision:",round(precision,2),"\tRecall:",round(recall,2),"\tF1 Score:",round(f1score,2))
              
        
        st.write("Gridsearch is used to find the best hyperparameters and a stratified K-fold cross validation is used to evaluate the model.")
        st.write("Use to get the best Optimized Hyperparameters :")
        if st.button("GridSearch RF"):
            n_estimators = [10, 100, 1000]
            max_features = ['sqrt', 'log2']
            # define grid search
            grid = dict(n_estimators=n_estimators,max_features=max_features)
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            grid_search = GridSearchCV(estimator=rf, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
            grid_result = grid_search.fit(X, y)
            # summarize results
            st.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))




        fig=px.imshow(confusion,text_auto=True,labels=dict(x="Actual Label", y="Predicted Label"),x=['Heart Disease','No Heart Disease'],y=['Heart Disease','No Heart Disease'])
        fig.update_layout(title_text='Confusion Matrix', title_x=0.5)
        st.plotly_chart(fig)
        
        fig=px.line(fpr,tpr,labels=dict(x="False Positive Rate", y="True positive Rate"))
        fig.update_layout(title_text='ROC Curve',title_x=0.5)
        st.plotly_chart(fig)


    with tab3:
        with st.expander("K-Nearest Neighbors"):
            st.write("""

            KNN (K-Nearest Neighbor) is a simple supervised classification algorithm we can use to assign a class to new data point. It can be used for regression as well, KNN does not make any assumptions on the data distribution, hence it is non-parametric. It keeps all the training data to make future predictions by computing the similarity between an input sample and each training instance.

KNN can be summarized as below:

Computes the distance between the new data point with every training example.
For computing the distance measures such as Euclidean distance, Hamming distance or Manhattan distance will be used.
Model picks K entries in the database which are closest to the new data point.
Then it does the majority vote i.e the most common class/label among those K entries will be the class of the new data point.
            
            """
            
            
            )

            st.image("https://miro.medium.com/max/1174/1*hncgU7vWLBsRvc8WJhxlkQ.webp")

        
        col1,col2,col3= st.columns(3)
        with col3:
            weight=st.selectbox("Weight Function",('uniform','distance'))
        with col1:
           algorithm=st.selectbox("Algorithm",('auto','ball_tree','kd_tree','brute'))
        
        with col2:
            neighbors=st.slider("n_neighbors",1,20,5)

        knn=KNeighborsClassifier(n_neighbors=neighbors,weights=weight, algorithm=algorithm)
        knn.fit(x_train, y_train)
        pred= knn.predict(x_test)
        fpr, tpr, _ =roc_curve(y_test,pred)
        
        accuracyscore=accuracy_score(y_test,pred)
        confusion=confusion_matrix(y_test,pred)
        f1score=f1_score(y_test,pred)
        precision=precision_score(y_test,pred)
        recall=recall_score(y_test,pred)
        st.write("#### Accuracy:",round(accuracyscore,2),   "\tPrecision:",round(precision,2),"\tRecall:",round(recall,2),"\tF1 Score:",round(f1score,2))
        
        st.write("Gridsearch is used to find the best hyperparameters and a stratified K-fold cross validation is used to evaluate the model.")
        st.write("Use to get the best Optimized Hyperparameters :")
        if st.button("GridSearch KNN"):
            n_neighbors = range(1, 21, 2)
            weights = ['uniform', 'distance']
            metric = ['euclidean', 'manhattan', 'minkowski']
            # define grid search
            grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            grid_search = GridSearchCV(estimator=knn, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
            grid_result = grid_search.fit(X, y)
            # summarize results
            st.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


        fig=px.imshow(confusion,text_auto=True,labels=dict(x="Actual Label", y="Predicted Label"),x=['Heart Disease','No Heart Disease'],y=['Heart Disease','No Heart Disease'])
        fig.update_layout(title_text='Confusion Matrix', title_x=0.5)
        st.plotly_chart(fig)
        
        fig=px.line(fpr,tpr,labels=dict(x="False Positive Rate", y="True positive Rate"))
        fig.update_layout(title_text='ROC Curve',title_x=0.5)
        st.plotly_chart(fig)



    with tab4:
        with st.expander("Support Vector Machine"):
            st.write("""
            The objective of the support vector machine algorithm is to find a hyperplane in an N dimensional space(the number of features) that distinctly classifies the data points.

            """)

            st.image("https://miro.medium.com/max/600/0*0o8xIA4k3gXUDCFU.webp")
            st.image("https://miro.medium.com/max/600/0*0o8xIA4k3gXUDCFU.webp")
        
        col1,col2,col3= st.columns(3)
        with col1:
            kernel=st.selectbox("Kernel",('linear','poly','rbf','sigmoid','precomputed'))
        with col2:
            rpar=st.slider("Regularization Parameter",0.01,1.0,1.0)
        
        with col3:
            gamma=st.selectbox("Gamma",('scale','auto'))

        svm=SVC(kernel=kernel,C=rpar,gamma=gamma,random_state=27)
        svm.fit(x_train, y_train)
        pred= svm.predict(x_test)
        
        fpr, tpr, _ =roc_curve(y_test,pred)
        
        accuracyscore=accuracy_score(y_test,pred)
        confusion=confusion_matrix(y_test,pred)
        f1score=f1_score(y_test,pred)
        precision=precision_score(y_test,pred)
        recall=recall_score(y_test,pred)
        st.write("#### Accuracy:",round(accuracyscore,2),   "\tPrecision:",round(precision,2),"\tRecall:",round(recall,2),"\tF1 Score:",round(f1score,2))
        
        st.write("Gridsearch is used to find the best hyperparameters and a stratified K-fold cross validation is used to evaluate the model.")
        st.write("Use to get the best Optimized Hyperparameters :")
        if st.button("GridSearch SVM"):
            kernel = ['poly', 'rbf', 'sigmoid']
            C = [50, 10, 1.0, 0.1, 0.01]
            gamma = ['scale']
            # define grid search
            grid = dict(kernel=kernel,C=C,gamma=gamma)
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            grid_search = GridSearchCV(estimator=svm, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
            grid_result = grid_search.fit(X, y)
            st.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        
        fig=px.imshow(confusion,text_auto=True,labels=dict(x="Actual Label", y="Predicted Label"),x=['Heart Disease','No Heart Disease'],y=['Heart Disease','No Heart Disease'])
        fig.update_layout(title_text='Confusion Matrix', title_x=0.5)
        st.plotly_chart(fig)
        
        fig=px.line(fpr,tpr,labels=dict(x="False Positive Rate", y="True positive Rate"))
        fig.update_layout(title_text='ROC Curve', title_x=0.5,xaxis_range=[0,1],yaxis_range=[0,1])
        st.plotly_chart(fig)
              
        
       

  


st.sidebar.write("The problem with heart disease is that the first symptom is often fatal. :broken_heart:")
from PIL import Image
#ima=Image.open("C:/Users/visha/Downloads/n0719b16207259263423.jpg")
st.sidebar.image("https://domf5oio6qrcr.cloudfront.net/medialibrary/5648/n0719b16207259263423.jpg")

page=st.sidebar.radio(' ',['EDA & Visualization',"ML Classification Models"])



if page=='EDA & Visualization':
       Introduction()
if page=='Data':
       Data()
if page=='Demographics(Categorical)':
       Demographics()
if page=='Demographics(Numerical)':
       Page2()
if page=='Insights&Summary':
       Page3()
if page=='ML Classification Models':
       Page4()




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
from os import path
import streamlit as st
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import sklearn
from PIL import Image
import warnings
from sklearn.neighbors import KNeighborsClassifier  

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler,OrdinalEncoder,MinMaxScaler,MaxAbsScaler,RobustScaler,normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,f1_score,precision_score,recall_score,roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV


st.sidebar.write("# **Heart Disease Prediction Models**")



warnings.filterwarnings('ignore')

sns.set_style("whitegrid")

colors6 = sns.color_palette(['#1337f5', '#E80000', '#0f1e41', '#fd523e', '#404e5c', '#c9bbaa'], 6)
colors2 = sns.color_palette(['#1337f5', '#E80000'], 2)
colors1 = sns.color_palette(['#1337f5'], 1)

df = pd.read_csv("heart.csv")

st.set_option('deprecation.showPyplotGlobalUse', False)

def show_relation(col, according_to, type_='dis'):
  plt.figure(figsize=(15,7));

  if type_=='dis':
    sns.displot(data=df, x=col, hue=according_to, kind='kde', palette=colors2);
  elif type_=='count':
    if according_to != None:
      perc = df.groupby(col)[according_to].value_counts(normalize=True).reset_index(name='Percentage')
      sns.barplot(data=perc, x=col,y='Percentage', hue=according_to, palette=colors6, order=df[col].value_counts().index);
    else:
      sns.countplot(data=df, x=col, hue=according_to, palette=colors1, order=df[col].value_counts().index);

  if according_to==None:
    plt.title(f'{col}');
  else: 
    plt.title(f'{col} according to {according_to}');
    
    
def generate_colors(num):
    colors = []
    lst = list('ABCDEF0123456789')

    for i in range(num):
        colors.append('#'+''.join(np.random.choice(lst, 6)))
        
    return colors


def Introduction():
    st.write("# **Indicators of Heart Disease**")
    st.write("## **Introduction**")
    st.write("According to the world health organization, Cardiovascular diseases (CVDs) are the leading cause of death globally. In 2019 alone, around 17.9 million people died from CVDs. Of these deaths, 85% of them were due to heart diseases. There are many factors that play a role in increasing the risk of heart disease. Identifying these factors and their impact is paramount in the field of healthcare. Identifying patients who are at greater risk enables medical professionals to respond quickly and efficiently, saving more lives.")
     #st.selectbox()

    st.write("In this project we will delve deep into the causes of heart disease and its relations with other health indicators, drawing insights and exploring the data in order to get a better picture of the leading causes.")
    from PIL import Image
    #image=Image.open("C:/Users/visha/Downloads/heart.jpg")
    st.image("https://www.kaggleusercontent.com/kf/107235006/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..4QLPzVXFb5D2NBoGfSVkUQ.-5jgWDnaJZe58QcPIucHKlK_9X_l-cHqjAXKxLTZByhMWa45VO_r3D-Lq8XqId_XNhophAnYhkx8YM0zu-6I2ic5MM7QPNY0pXC8fGiQNPH3Oz1yvWJ8kfjfNJhdE07SQFkJmwK9K3bOkLgez_6lwZo--m1PXppvAhrwCjO_FjEUkIWaN9NISnrLFSSOxevPvDDbiEcWq7slOVxw6vQR73fqZElv9ji-JxhAe4hqT2FSgutm39WIyPytZViP-SOKZKU6uabxLihsnVtH2pJaFxmJL0IBSwSlOVlgPIfkKY6q0tPiVxkmAFHy6_tHeLJSKj6uKQbgC6pj7n-c1SUQflJAk7kVDg_NlNw4oHG-MW-sbHy9aNtbShYhMb8GhGV6y_newdS52Ou5bjAEbdFHz6xyWVR81KsW1pn-xZviOOUoEQXuoFVdkAtMiS90EG5kphGUFJDwB6V8vtayIaqBeWibeBs9tWcn6QUU6f2dIkjuvpyrnw4Z3ooLAtN78VWxCswl4VNtXWgyo2UvmLzgZHaDQcLjbOd_mgDPqFSTnNQdBcqTNVZqYDNgt4gQypse-SAZT-cVbxidSDAYOr9QncfyCcXnchIjrWNJhlB-X2T5y9UkmrxBn4uuoFXXZW-U26y5RVHg23SyWYTgSiwj-uam-dilLW5j_GdzTPDcNwnAwUfTYBVM-bFjJ4gqstVj.7M0xFjCzQ-GDZaHBwPCQxg/__results___files/__results___31_0.png")

    if st.button("Glimpse of Heart Data"):
        st.write(df.head())

    fig=plt.figure(figsize=(15,7));
    plt.title('HeartDisease Count');
    sns.countplot(data=df, x='HeartDisease', palette=colors2, order=df['HeartDisease'].value_counts().index);
    st.pyplot(fig)



     ####




    st.write("**About the Dataset:**")
    st.write("""
     The dataset contains 320K rows and 18 columns. It is a cleaned, smaller version of the 2020 annual CDC (Centers for Disease Control and Prevention) survey data of 400k adults. For each patient (row), it contains the health status of that individual. The data was collected in the form of surveys conducted over the phone. Each year, the CDC calls around 400K U.S residents and asks them about their health status, with the vast majority of questions being yes or no questions. Below is a description of the features collected for each patient:
     
     
     """)

    if st.button("Glimpse of Data Columns"):
        from PIL import Image
        #im=Image.open("https://drive.google.com/file/d/1IPpgndT1pAHyV9AhD8nw49DmDRN9ekqC/view?usp=share_link")
        st.image("https://drive.google.com/file/d/1IPpgndT1pAHyV9AhD8nw49DmDRN9ekqC/view?usp=share_link")


    st.write("**Initial Data Assessment of Numerical Values:**")
    categorical=list(set(df.columns)-set(['BMI','SleepTime','PhysicalHealth','MentalHealth']))
    numerical=list(['BMI','SleepTime','PhysicalHealth','MentalHealth'])

    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches(15, 9)
    fig.tight_layout(pad=5)
    k = 0
    numerical_variables=numerical
    for num_var in numerical_variables:
         k = int(numerical_variables.index(num_var)/2)
         sns.distplot(df[num_var], ax=axes[k, numerical_variables.index(num_var)%2])

    plt.plot()
    plt.show()
    st.pyplot(fig)
    st.write("""
    - BMI is skewed.  
    - Physical Health & Mental Health are severely imbalanced due to number of zeroes.
    - SleepTime is normally distributed.
                """)



         #################
    obj_cols = df.select_dtypes(include='object').columns[1:]
    num_cols = df.select_dtypes(exclude='object').columns


    fig=plt.figure(figsize=(20, 40))
    for i in range(len(obj_cols)):
      plt.subplot(7, 2, i+1)

      if(df[obj_cols[i]].nunique() < 3):
        ax = sns.countplot(data=df, x=obj_cols[i], palette=colors2, order=df[obj_cols[i]].value_counts().index[:6])
      else:
        ax = sns.countplot(data=df, x=obj_cols[i], palette=colors6, order=df[obj_cols[i]].value_counts().index[:6])

  
      plt.title(f'{obj_cols[i]}', fontsize=15, fontweight='bold', color='brown')
      plt.subplots_adjust(hspace=0.5)

      for p in ax.patches:
        height = p.get_height() 
        width = p.get_width()
        percent = height/len(df)

        ax.text(x=p.get_x()+width/2, y=height+2, s=format(percent, ".2%"), fontsize=12, ha='center', weight='bold')
    st.pyplot(fig)

    st.write(""""
    - Most of people in our data are white and have no diabetic
    - A litle of them who have asthma, kidney disease and skin cancer.    
    """)

  
    

    obj_cols = df.select_dtypes(include='object').columns[1:]
    num_cols = df.select_dtypes(exclude='object').columns

    
    features=st.selectbox("Feature Selection",obj_cols)


    


    obj_cols = df.select_dtypes(include='object').columns[1:]
    num_cols = df.select_dtypes(exclude='object').columns
    features=st.selectbox("Feature Selection",num_cols)
    if features=='BMI':
        st.write("Is the BMI of heart disease patients different?")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        fig=show_relation(num_cols[0], 'HeartDisease')
        st.pyplot(fig)

        fig=plt.figure(figsize=(16, 6), dpi=80)

        sns.boxplot(data=df, x='BMI', y='HeartDisease', saturation=0.4,width=0.15, boxprops={'zorder': 2},showfliers = False, whis=0,  palette=colors2);
        sns.violinplot(data=df, x='BMI', y='HeartDisease',inner='quartile', palette=colors2);
        st.pyplot(fig)

        st.write("""
        - the both distributions are normal distriubtions and in the same range which is from 10 t0 90 approximately.
        - The BMI distribution of individuals who suffer from heart disease is slightly shifted towards higher values in comparison to the distribution of those who don't. 
        """)

        st.write("Does BMI differ across diseases?")

        fig, ax = plt.subplots(figsize = (14,6))
        sns.kdeplot(df[df["HeartDisease"]=='Yes']["BMI"], alpha=1,shade = False, color=colors6[0], label="HeartDisease", ax = ax)
        sns.kdeplot(df[df["KidneyDisease"]=='Yes']["BMI"], alpha=1,shade = False, color=colors6[1], label="KidneyDisease", ax = ax)
        sns.kdeplot(df[df["SkinCancer"]=='Yes']["BMI"], alpha=1,shade = False, color=colors6[2], label="SkinCancer", ax = ax)
        sns.kdeplot(df[df["Asthma"]=='Yes']["BMI"], alpha=1,shade = False, color=colors6[3], label="Asthma", ax = ax)
        sns.kdeplot(df[df["Stroke"]=='Yes']["BMI"], alpha=1,shade = False, color=colors6[4], label="Stroke", ax = ax)
        sns.kdeplot(df[df["Diabetic"]=='Yes']["BMI"], alpha=1,shade = False, color=colors6[5], label="Diabetic", ax = ax)


        ax.set_xlabel("BMI")
        ax.set_ylabel("Frequency")
        ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
        plt.show()
        st.pyplot(fig)

    if features=='PhysicalHealth':
        st.pyplot(show_relation(num_cols[1], 'HeartDisease'))


        fig, ax = plt.subplots(figsize = (14,6))
        sns.kdeplot(df[df["HeartDisease"]=='Yes']["PhysicalHealth"], alpha=1,shade = False, color=colors6[0], label="HeartDisease", ax = ax)
        sns.kdeplot(df[df["KidneyDisease"]=='Yes']["PhysicalHealth"], alpha=1,shade = False, color=colors6[1], label="KidneyDisease", ax = ax)
        sns.kdeplot(df[df["SkinCancer"]=='Yes']["PhysicalHealth"], alpha=1,shade = False, color=colors6[2], label="SkinCancer", ax = ax)
        sns.kdeplot(df[df["Asthma"]=='Yes']["PhysicalHealth"], alpha=1,shade = False, color=colors6[3], label="Asthma", ax = ax)
        sns.kdeplot(df[df["Stroke"]=='Yes']["PhysicalHealth"], alpha=1,shade = False, color=colors6[4], label="Stroke", ax = ax)
        sns.kdeplot(df[df["Diabetic"]=='Yes']["PhysicalHealth"], alpha=1,shade = False, color=colors6[5], label="Diabetic", ax = ax)


        ax.set_xlabel("PhysicalHealth")
        ax.set_ylabel("Frequency")
        ax.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
        plt.show()
        st.pyplot(fig)


    if features=='MentalHealth':
        st.write("Are heart disease patients more mentally unwell?")




        fig=show_relation(num_cols[2], 'HeartDisease')
        st.pyplot(fig)



        



    if features=='SleepTime':

        st.write("Distribution of sleep time among heart disease patients")

        st.pyplot(show_relation(num_cols[3], 'HeartDisease'))

        relative = df.groupby('HeartDisease').SleepTime.value_counts(normalize=True).reset_index(name='Percentage')

        fig=plt.figure(figsize=(16, 6), dpi=80)
        ax = sns.barplot(data=relative, x='SleepTime', y='Percentage', hue='HeartDisease', palette=colors2);

        ax.set_title("Percentage of Sleep Times by Heart Disease");

        st.pyplot(fig)

        st.write("""
        - Abnormal sleeep duration is more prevalent in heart disease patients.
        - Higher percentages of sleep less than 6 hours or more than 9 hours have heart disease.
        """)




    st.write("## **Summary**")
    st.write("""
    - Around 9 among 100 individuals suffer from heart disease
    - The BMI of heart disease patients is slightly higher than that of healthy individuals.
    - The older the individual, the more susceptible they are to heart disease.
    - A lot more people who suffer from heart disease say they have poor or fair health compared to those who don't
    - 79% of healthy individuals have been physically active in the past 30 days, compared to 64% in heart disease patients.
    - Abnormal sleeep duration is more prevalent in heart disease patients
    - people who smoke suffer from heart disease
    - Those who have suffered from kidney disease & Asthma are at a sginificantly higher risk of heart disease.
    - Mental health, sleep duration, and physical health are similar among people who suffer from different diseases.
    - Having a stroke is highly correlacted with heart disease.
    




    """)

def Page4():
    heart=pd.read_csv("heart_2020_ml.csv")
    
    st.write("## **Data Pre-Processing & Feature Engineering**")
    features=st.multiselect("Number of Features",['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke',
       'PhysicalHealth', 'MentalHealth', 'DiffWalking', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'SleepTime',
       'Asthma', 'KidneyDisease', 'SkinCancer'],['BMI','GenHealth','SleepTime','PhysicalHealth', 'MentalHealth','Smoking','AlcoholDrinking','DiffWalking'])
    X=np.array(heart[features])
    y=np.array(heart[['HeartDisease']])
    n=st.slider("Test Size",0.0,1.0,0.3)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = n, random_state = 0)
    
    

    scaling=st.selectbox("Scaling",('No Scaling','Absolute Maximum Scaling','Min-Max Scaling','Normalization','Standard Scaling','Robust Scaling'))
    if scaling=='No Scaling':
        x_train = x_train
        x_test =x_test

    
    
    if scaling=='Absolute Maximum Scaling':
        asc = MaxAbsScaler()
        x_train = asc.fit_transform(x_train)
        x_test = asc.transform(x_test)

    if scaling=='Min-Max Scaling':
        sc = MinMaxScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)
    if scaling=='Normalization':
        x_train = normalize(x_train)
        x_test = normalize(x_test)

    if scaling=='Standard Scaling':
        ssc = StandardScaler()
        x_train = ssc.fit_transform(x_train)
        x_test = ssc.transform(x_test)
    if scaling=='Robust Scaling':
        rsc = RobustScaler()
        x_train = rsc.fit_transform(x_train)
        x_test = rsc.transform(x_test)







    st.write("# **Machine Learning Models to Classify Whether a Person is having Heart Disease or not**")
    tab1,tab2,tab3,tab4=st.tabs([' Logistic Regression','Random Forest','K-Nearest Neighbor','Support Vector Machine'])

    

    with tab1:

        with st.expander("Logistic Regression"):
                st.write(              """
                    Logistic Regression was used in the biological sciences in early twentieth century. It was then used in many social science applications. Logistic Regression is used when the dependent variable(target) is categorical.

For example,

To predict whether an email is spam (1) or (0)
Whether the tumor is malignant (1) or not (0)
Consider a scenario where we need to classify whether an email is spam or not. If we use linear regression for this problem, there is a need for setting up a threshold based on which classification can be done. Say if the actual class is malignant, predicted continuous value 0.4 and the threshold value is 0.5, the data point will be classified as not malignant which can lead to serious consequence in real time.
                 
                    """
                    
                    )
                st.image("https://miro.medium.com/max/1400/1*RqXFpiNGwdiKBWyLJc_E7g.webp")

        

        
        
        





                    
            
            

        col1,col2,col3= st.columns(3)
        with col1:
            solvers=st.selectbox("Select Solver",('newton-cg', 'lbfgs', 'liblinear'))
        with col2:
            c_values=st.slider("Alpha",0.001,1.0,1.0)
        with col3:
            pen=st.selectbox("Penalty",('none','l1','l2','elasticnet'))
            
     

        lr=LogisticRegression(penalty=pen,C=c_values,solver=solvers)
        lr.fit(x_train, y_train)
        pred= lr.predict(x_test)
        lr_probs = lr.predict_proba(x_test)[::,1]
        fpr, tpr, _ =roc_curve(y_test,lr_probs)
        
        accuracyscore=accuracy_score(y_test,pred)
        confusion=confusion_matrix(y_test,pred)
        f1score=f1_score(y_test,pred)
        precision=precision_score(y_test,pred)
        recall=recall_score(y_test,pred)
        st.write("#### Accuracy:",round(accuracyscore,2),   "\t Precision:",round(precision,2),"\t Recall:",round(recall,2),"\t F1 Score:",round(f1score,2))
        report = classification_report(y_test,pred, output_dict=True)

        
        
        st.write("Gridsearch is used to find the best hyperparameters and a stratified K-fold cross validation is used to evaluate the model.")
        st.write("Use to get the best Optimized Hyperparameters :")
        if st.button("GridSearch LR"):
            solvers = ['newton-cg', 'lbfgs', 'liblinear']
            penalty = ['l2']
            c_values = [1.0, 0.1, 0.01]
            # define grid search
            grid = dict(solver=solvers,penalty=penalty,C=c_values)
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            grid_search = GridSearchCV(estimator=lr, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
            grid_result = grid_search.fit(X,y)
            # summarize results
            st.write("Accuracy:",round(accuracyscore,2),   "\tPrecision:",round(precision,2),"\tRecall:",round(recall,2),"\tF1 Score:",round(f1score,2))
        

        fig=px.imshow(confusion,text_auto=True,labels=dict(x="Actual Label", y="Predicted Label"),x=['Heart Disease','No Heart Disease'],y=['Heart Disease','No Heart Disease'])
        fig.update_layout(title_text='Confusion Matrix', title_x=0.5)
        st.plotly_chart(fig)
        

        fig=px.line(fpr,tpr,labels=dict(x="False Positive Rate", y="True positive Rate"))

        fig.update_layout(title_text='ROC Curve',title_x=0.5)
        st.plotly_chart(fig)

               
        
   
          
        

    with tab2:
        with st.expander("Random Forest"):
            st.write("""
            Random forest, like its name implies, consists of a large number of individual decision trees that operate as an ensemble. Each individual tree in the random forest spits out a class prediction and the class with the most votes becomes our model's prediction (see figure below).
            
            
            """)
            st.image("https://miro.medium.com/max/1052/1*VHDtVaDPNepRglIAv72BFg.webp")

        
        col1,col2,col3= st.columns(3)
        with col1:
            estimators=st.slider("Estimators",0,1000,100)
        with col2:
           criter=st.selectbox("Split Criterion",('gini', 'entropy', 'log_loss'))
        
        with col3:
            depth=st.slider("Maximum Depth",1,10,5)

        rf=RandomForestClassifier(n_estimators=estimators,criterion=criter, max_depth=depth)
        rf.fit(x_train, y_train)
        pred= rf.predict(x_test)
        fpr, tpr, _ =roc_curve(y_test,pred)
        
        accuracyscore=accuracy_score(y_test,pred)
        confusion=confusion_matrix(y_test,pred)
        f1score=f1_score(y_test,pred)
        precision=precision_score(y_test,pred)
        recall=recall_score(y_test,pred)
        st.write("#### Accuracy:",round(accuracyscore,2),   "\tPrecision:",round(precision,2),"\tRecall:",round(recall,2),"\tF1 Score:",round(f1score,2))
              
        
        st.write("Gridsearch is used to find the best hyperparameters and a stratified K-fold cross validation is used to evaluate the model.")
        st.write("Use to get the best Optimized Hyperparameters :")
        if st.button("GridSearch RF"):
            n_estimators = [10, 100, 1000]
            max_features = ['sqrt', 'log2']
            # define grid search
            grid = dict(n_estimators=n_estimators,max_features=max_features)
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            grid_search = GridSearchCV(estimator=rf, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
            grid_result = grid_search.fit(X, y)
            # summarize results
            st.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))




        fig=px.imshow(confusion,text_auto=True,labels=dict(x="Actual Label", y="Predicted Label"),x=['Heart Disease','No Heart Disease'],y=['Heart Disease','No Heart Disease'])
        fig.update_layout(title_text='Confusion Matrix', title_x=0.5)
        st.plotly_chart(fig)
        
        fig=px.line(fpr,tpr,labels=dict(x="False Positive Rate", y="True positive Rate"))
        fig.update_layout(title_text='ROC Curve',title_x=0.5)
        st.plotly_chart(fig)


    with tab3:
        with st.expander("K-Nearest Neighbors"):
            st.write("""

            KNN (K-Nearest Neighbor) is a simple supervised classification algorithm we can use to assign a class to new data point. It can be used for regression as well, KNN does not make any assumptions on the data distribution, hence it is non-parametric. It keeps all the training data to make future predictions by computing the similarity between an input sample and each training instance.

KNN can be summarized as below:

Computes the distance between the new data point with every training example.
For computing the distance measures such as Euclidean distance, Hamming distance or Manhattan distance will be used.
Model picks K entries in the database which are closest to the new data point.
Then it does the majority vote i.e the most common class/label among those K entries will be the class of the new data point.
            
            """
            
            
            )

            st.image("https://miro.medium.com/max/1174/1*hncgU7vWLBsRvc8WJhxlkQ.webp")

        
        col1,col2,col3= st.columns(3)
        with col3:
            weight=st.selectbox("Weight Function",('uniform','distance'))
        with col1:
           algorithm=st.selectbox("Algorithm",('auto','ball_tree','kd_tree','brute'))
        
        with col2:
            neighbors=st.slider("n_neighbors",1,20,5)

        knn=KNeighborsClassifier(n_neighbors=neighbors,weights=weight, algorithm=algorithm)
        knn.fit(x_train, y_train)
        pred= knn.predict(x_test)
        fpr, tpr, _ =roc_curve(y_test,pred)
        
        accuracyscore=accuracy_score(y_test,pred)
        confusion=confusion_matrix(y_test,pred)
        f1score=f1_score(y_test,pred)
        precision=precision_score(y_test,pred)
        recall=recall_score(y_test,pred)
        st.write("#### Accuracy:",round(accuracyscore,2),   "\tPrecision:",round(precision,2),"\tRecall:",round(recall,2),"\tF1 Score:",round(f1score,2))
        
        st.write("Gridsearch is used to find the best hyperparameters and a stratified K-fold cross validation is used to evaluate the model.")
        st.write("Use to get the best Optimized Hyperparameters :")
        if st.button("GridSearch KNN"):
            n_neighbors = range(1, 21, 2)
            weights = ['uniform', 'distance']
            metric = ['euclidean', 'manhattan', 'minkowski']
            # define grid search
            grid = dict(n_neighbors=n_neighbors,weights=weights,metric=metric)
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            grid_search = GridSearchCV(estimator=knn, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
            grid_result = grid_search.fit(X, y)
            # summarize results
            st.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


        fig=px.imshow(confusion,text_auto=True,labels=dict(x="Actual Label", y="Predicted Label"),x=['Heart Disease','No Heart Disease'],y=['Heart Disease','No Heart Disease'])
        fig.update_layout(title_text='Confusion Matrix', title_x=0.5)
        st.plotly_chart(fig)
        
        fig=px.line(fpr,tpr,labels=dict(x="False Positive Rate", y="True positive Rate"))
        fig.update_layout(title_text='ROC Curve',title_x=0.5)
        st.plotly_chart(fig)



    with tab4:
        with st.expander("Support Vector Machine"):
            st.write("""
            The objective of the support vector machine algorithm is to find a hyperplane in an N dimensional space(the number of features) that distinctly classifies the data points.

            """)

            st.image("https://miro.medium.com/max/600/0*0o8xIA4k3gXUDCFU.webp")
            st.image("https://miro.medium.com/max/600/0*0o8xIA4k3gXUDCFU.webp")
        
        col1,col2,col3= st.columns(3)
        with col1:
            kernel=st.selectbox("Kernel",('linear','poly','rbf','sigmoid','precomputed'))
        with col2:
            rpar=st.slider("Regularization Parameter",0.01,1.0,1.0)
        
        with col3:
            gamma=st.selectbox("Gamma",('scale','auto'))

        svm=SVC(kernel=kernel,C=rpar,gamma=gamma,random_state=27)
        svm.fit(x_train, y_train)
        pred= svm.predict(x_test)
        
        fpr, tpr, _ =roc_curve(y_test,pred)
        
        accuracyscore=accuracy_score(y_test,pred)
        confusion=confusion_matrix(y_test,pred)
        f1score=f1_score(y_test,pred)
        precision=precision_score(y_test,pred)
        recall=recall_score(y_test,pred)
        st.write("#### Accuracy:",round(accuracyscore,2),   "\tPrecision:",round(precision,2),"\tRecall:",round(recall,2),"\tF1 Score:",round(f1score,2))
        
        st.write("Gridsearch is used to find the best hyperparameters and a stratified K-fold cross validation is used to evaluate the model.")
        st.write("Use to get the best Optimized Hyperparameters :")
        if st.button("GridSearch SVM"):
            kernel = ['poly', 'rbf', 'sigmoid']
            C = [50, 10, 1.0, 0.1, 0.01]
            gamma = ['scale']
            # define grid search
            grid = dict(kernel=kernel,C=C,gamma=gamma)
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            grid_search = GridSearchCV(estimator=svm, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
            grid_result = grid_search.fit(X, y)
            st.write("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        
        fig=px.imshow(confusion,text_auto=True,labels=dict(x="Actual Label", y="Predicted Label"),x=['Heart Disease','No Heart Disease'],y=['Heart Disease','No Heart Disease'])
        fig.update_layout(title_text='Confusion Matrix', title_x=0.5)
        st.plotly_chart(fig)
        
        fig=px.line(fpr,tpr,labels=dict(x="False Positive Rate", y="True positive Rate"))
        fig.update_layout(title_text='ROC Curve', title_x=0.5,xaxis_range=[0,1],yaxis_range=[0,1])
        st.plotly_chart(fig)
              
        
       

  


st.sidebar.write("The problem with heart disease is that the first symptom is often fatal. :broken_heart:")
from PIL import Image
#ima=Image.open("C:/Users/visha/Downloads/n0719b16207259263423.jpg")
st.sidebar.image("https://domf5oio6qrcr.cloudfront.net/medialibrary/5648/n0719b16207259263423.jpg")

page=st.sidebar.radio(' ',['EDA & Visualization',"ML Classification Models"])



if page=='EDA & Visualization':
       Introduction()
if page=='Data':
       Data()
if page=='Demographics(Categorical)':
       Demographics()
if page=='Demographics(Numerical)':
       Page2()
if page=='Insights&Summary':
       Page3()
if page=='ML Classification Models':
       Page4()




