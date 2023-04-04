
import streamlit as st
import pandas as pd
from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics as mt
import seaborn as sns
from sklearn.metrics import accuracy_score

def main():
    def _max_width_():
        max_width_str = f"max-width: 1000px;"
        st.markdown(
            f"""
        <style>
        .reportview-container .main .block-container{{
            {max_width_str}
        }}
        </style>
        """,
            unsafe_allow_html=True,
        )


    # Hide the Streamlit header and footer
    def hide_header_footer():
        hide_streamlit_style = """
                    <style>
                    footer {visibility: hidden;}
                    </style>
                    """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # increases the width of the text and tables/figures
    _max_width_()

    # hide the footer
    hide_header_footer()


### The first two lines of the code load an image and display it using the st.image function.

### The st.title function sets the title of the web application to "Mid Term Template - 01 Introduction Page 🧪".



### The st.sidebar function creates a sidebar and adds a header and a markdown line. The app_mode variable is created to hold the selected option from the dropdown menu.
st.sidebar.header("Dashboard")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox('Select Page',['Summary 🚀','Visualization 📊','Prediction 📈'])
select_dataset =  st.sidebar.selectbox('💾 Select Dataset',["Fatalites caused by Law Enforcement"])
df = pd.read_csv("df_police_fatalities_merged.csv")
if app_mode == 'Summary 🚀':
    st.title("Deaths Due to the use of Lethal Force by Law Enforcement 👮🏻‍♂️")
    st.subheader("01 Summary Page - Spotify Data Analysis 🚀")
    image_police = Image.open('images/police.jpeg')
    st.image(image_police, width=750)


    ### The select_dataset variable is created to hold the selected dataset from the dropdown menu. The pd.read_csv function is used to read the data from a CSV file.
    ### The st.markdown function is used to display some text and headings on the web application.
    st.markdown("### 00 - Dataset")

    ### The st.number_input function creates a widget that allows the user to input a number. The st.radio function creates a radio button widget that allows the user to select either "Head" or "Tail".
    num = st.number_input('No. of Rows', 5, 15)
    head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))
    if head == 'Head':
    ### the st.dataframe function displays the data frame.
        st.dataframe(df.head(num))
    else:
        st.dataframe(df.tail(num))


    st.markdown("Number of rows and columns helps us to determine how large the dataset is.")
    ### The st.text and st.write functions display the shape of the data frame and some information about the variables in the data set.
    st.text('(Rows,Columns)')
    st.write(df.shape)



    st.markdown("##### variables ➡️")
    st.markdown(" **UID**: the unique identifier used to track-and-trace the people in the system")
    st.markdown(" **Name**: the names of the people who were killed due to use of leathal force by law enforcement")
    st.markdown(" **Age**: the age of the deceased")
    st.markdown(" **Gender**: the gender of the deceased (Male or Female)")
    st.markdown(" **Race**: the race of the deceased (Asian, Black, Hispanic, Native, Other, White)")
    st.markdown(" **Date**: the date the person was murdered by law enforcement")
    st.markdown(" **City**: the city that the person died")
    st.markdown(" **State**: the state that the person died")
    st.markdown(" **Manner of Death**: the method used by law enforcement to kill the person (Shot,Tasered, Both)")
    st.markdown(" **Armed**: the weapon carried by the person who was killed by law enforcement (Gun, Knife, Unarmed, Vehicle, Toy Weapon, Machete, Sword)")
    st.markdown(" **Mental Illness**: the deceased either had or didn't have a mental illness (checked box: positive for mental illness and empty box: negative for mental illness)")
    st.markdown(" **Flee**: the deceased either fled the scene or stayed on the scene (checked box: fled the scene and empty box: stayed on the scene)")
    st.markdown(" **State Code**: the two-letter alphabetic code that is unique to each US state ")
    st.markdown(" **Pop Est 2014**: the established population in the state the person was killed in 2014")



    ### The st.markdown and st.dataframe functions display the descriptive statistics of the data set.
    st.markdown("### 01 - Description")
    st.dataframe(df.describe())


    ### The st.markdown, st.write, and st.warning functions are used to display information about the missing values in the data set.
    st.markdown("### 02 - Missing Values")
    st.markdown("Missing values are known as null or NaN values. Missing data tends to **introduce bias that leads to misleading results.**")
    dfnull = df.isnull().sum()/len(df)*100
    totalmiss = dfnull.sum().round(2)
    st.write("Percentage of total missing values:",totalmiss)
    st.write(dfnull)
    if totalmiss <= 30:
        st.success("Looks good! as we have less then 30 percent of missing values.")
    else:
        st.warning("Poor data quality due to greater than 30 percent of missing value.")
    st.markdown(" > Theoretically, 25 to 30 percent is the maximum missing values are allowed, there's no hard and fast rule to decide this threshold. It can vary from problem to problem.")



    st.markdown("### 03 - Completeness")
    st.markdown(" Completeness is defined as the ratio of non-missing values to total records in dataset.") 
    # st.write("Total data length:", len(df))
    nonmissing = (df.notnull().sum().round(2))
    completeness= round(sum(nonmissing)/len(df),2)
    st.write("Completeness ratio:",completeness)
    st.write(nonmissing)
    if completeness >= 0.80:
        st.success("Looks good! as we have completeness ratio greater than 0.85.")    
    else:
        st.success("Poor data quality due to low completeness ratio( less than 0.85).")

if app_mode == 'Visualization 📊':
    st.subheader("Visualization 📊")

    st.markdown("[![Foo](https://i.postimg.cc/1XxrpRnQ/looker.png)](https://lookerstudio.google.com/u/0/reporting/8068b0af-4b2f-48a6-aeba-33a6fdb1f87c)")


if app_mode == 'Prediction 📈':



    ### The st.title() function sets the title of the Streamlit application to "Mid Term Template - 03 Prediction Page 🧪".
    st.subheader("03 Prediction Page 🧪")

    ### The st.title() function sets the title of the Streamlit application to "Mid Term Template - 03 Prediction Page 🧪".
    st.title(" Prediction Page 👮🏻‍♀️")



    ### read csv files
    df = pd.read_csv('df_police_fatalities_merged.csv')





    ###factorize binary data within the df

    ###gender encoding
    df['Gender'] = df['Gender'].factorize()[0]

    ### race encoding
    df['Race'] = df['Race'].factorize()[0]

    ### statecode encoding
    df['stateCode'] = df['stateCode'].factorize()[0]

    ### armed encoding
    df['Armed'] = df['Armed'].factorize()[0]

    ### mentalilness
    df['Mental_illness'] = df['Mental_illness'].factorize()[0]

    ### flee
    df['Flee'] = df['Flee'].factorize()[0]

    ### Manner of death 
    df['Manner_of_death'] = df['Manner_of_death'].factorize()[0]

    ### remove commas in population so data could be more readable
    df[" popEst2014 "] = pd.to_numeric(df[" popEst2014 "].str.replace(",",""))

    df=df.dropna()


    ### The st.sidebar.selectbox() function creates a dropdown menu in the sidebar that allows users to select the target variable to predict.
    list_variables = df.columns
    select_variable =  st.sidebar.selectbox('🎯 Select Variable to Predict',list_variables)

    ### The st.sidebar.number_input() function creates a number input widget in the sidebar that allows users to select the size of the training set.

    new_df= df.drop(labels=select_variable, axis=1)  #axis=1 means we drop data by columns
    list_var = new_df.columns

    ### The st.multiselect() function creates a multiselect dropdown menu that allows users to select the explanatory variables.
    output_multi = st.multiselect("Select Explanatory Variables", list_var,default= ["Race","Gender", "Flee","stateCode"])
    new_df2 = new_df[output_multi]
    X =  new_df2
    y = df["Mental_illness"]
    train_size = st.sidebar.number_input("Train Set Size", min_value=0.00, step=0.01, max_value=1.00, value=0.70)

    ### The train_test_split() function splits the data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = train_size)

    ### The LogisticRegression() function creates a logistic regression model.
    lm = LogisticRegression()

    ### The lm.fit() function fits the linear regression model to the training data.
    lm.fit(X_train,y_train)

    ###The lm.predict() function generates predictions for the testing data.
    prediction = lm.predict(X_test)

    ###Calculate score
    score = accuracy_score(y_test, prediction)



    ### The st.columns() function creates two columns to display the feature columns and target column.
    col1,col2 = st.columns(2)
    col1.subheader("Feature Columns top 25")
    col1.write(X.head(25))
    col2.subheader("Target Column top 25")
    col2.write(y.head(25))

    ### The st.subheader() function creates a subheading for the results section.
    st.subheader('🎯 Results')
    ###print score
    st.write("🎯 Accuracy %",score * 100)


st.markdown(" ")
st.markdown("##### 💻 **App Contributors: Sydney Barragan, Aissatou Simaha, Tarzhay Robasson** ")

st.markdown(f"#####  Link to Project Website [here]({'https://github.com/NYU-DS-4-Everyone'}) 🚀 ")
st.markdown(f"#####  Feel free to contribute to the app and give a ⭐️")