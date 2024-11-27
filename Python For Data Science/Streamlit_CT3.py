import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


@st.cache_data
def load_data():
# Load data
    df = pd.read_csv('Loan_Sanction_DataSet.csv').drop(['Loan_ID'], axis=1)
    return df

def univariant_graph(feature,graph_type,color):

    df=load_data()
    st.subheader('Univariate Analysis -'+feature+' Column')
    
    if(graph_type == 'Box Plot' ):
        plt.figure(figsize=(10, 6))
        st.subheader(f'Box Plot for {feature}')
        sns.boxplot(x=feature, data=df,fill=True,color=color)
        st.pyplot()
        
    elif(graph_type == 'Grouped Box Plot'):
        st.subheader(f'Grouped Box Plot for all Features')
        df.boxplot(figsize=(20, 10))
        st.pyplot()

    elif(graph_type == 'Histogram'):
         st.subheader(f'Histogram for {feature}')
         num_bins = st.sidebar.slider("Number of Bins (for Histogram)", min_value=5, max_value=50, value=20)
         sns.histplot(df[feature], bins=num_bins, kde=True,color=color)
         plt.xlabel(feature)
         plt.ylabel('Frequency')
         st.pyplot()

    elif (graph_type == 'Count Plot'):
         st.subheader(f'Count Plot for {feature}')
         sns.countplot(x=feature,data=df,color=color)
         st.pyplot()

    elif(graph_type == 'Pie Chart'):  
         status_distribution = df[feature].value_counts()
         fig, ax = plt.subplots(figsize=(2.5,2.5))
         ax.pie(status_distribution, labels=status_distribution.index, autopct='%1.1f%%', startangle=90)
         ax.set_title(f'Pie Chart for {feature} Distribution')
         st.pyplot(fig)

    else :
        pass

def binvariant_graph(x,y,graph_type,color,col):

    df=load_data()
    st.subheader(f'Bivariate Analysis of  {x} and {y} ')
    
    if(graph_type == 'Scatter Plot' ):
        st.subheader(f'Scatter Plot for {x} Vs {y}')
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x,y=y,data=df,color=color)
        plt.xlabel(x)
        plt.ylabel(y)
        st.pyplot()
    
    elif(graph_type == 'Pair Plot'):
        st.subheader(f'Pair Plot of all Features ')
        sns.pairplot(df[col])
        plt.suptitle('Pairwise Scatter Plots of Numerical Variables', y=1.02)
        st.pyplot()
    
    elif(graph_type == 'Bar Plot'):
       st.subheader(f'Bar Plot : {x} Vs {y}')
       plt.suptitle('Bivariate Analysis: 2D Bar PLot', y=1.02)
       sns.barplot(x=x, y=y, data=df) 
       st.pyplot()

    elif(graph_type == 'Count Plot'):
        st.subheader(f'Bar PLot on Categorical Variables {x} VS {y}')
        plt.figure(figsize=(10, 6))
        sns.countplot(x=x, hue=y, data=df)
        st.pyplot()
    
    else:
        pass

def multivariant_graph(x,y,z,graph_type,color,col):

    df=load_data()
    st.subheader(f'Multivariate Analysis of  {x} and {y}  on {z} ')
    
    if(graph_type == 'Scatter Plot' ):
        st.subheader(f'Scatter Plot for {x} Vs {y} On {z}')
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x,y=y,hue=z,data=df)
        plt.xlabel(x)
        plt.ylabel(y)
        st.pyplot()
    
    elif(graph_type == 'Pair Plot'):
        st.subheader(f'Pairwise Scatter Plots of Numerical Variables on {z}')
        sns.pairplot(df,vars=col,hue=z)
        st.pyplot()
    
    elif(graph_type == 'Heat Map'):
       
       correlation_matrix = df[col].corr()
       plt.figure(figsize=(10, 8))
       sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
       plt.title('Correlation Matrix Heatmap')
       plt.show()
       st.pyplot()

    else:
        pass


def main():
          
    #Title page
    #st.set_option('deprecation.showPyplotGlobalUse', False)
    st.set_page_config(page_title="Loan Approval Analysis", layout="wide",initial_sidebar_state="expanded")
    st.title('Loan Approval Analysis')
    dynamic_content = st.empty()

    # Streamlit App
    df=load_data()
    dynamic_content.write(df)

    #Numerical and Catergorical columns 
    numeric_columns = df.select_dtypes(include=['int', 'float']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

# Sidebar with options
    analysis_type = st.sidebar.selectbox('Select Analysis Type', ['Select One', 'Univariate', 'Bivariate', 'Multivariate'])
    
# Univariate Visualization
    if analysis_type == 'Univariate':
        dynamic_content.empty()

        variable_type = st.sidebar.selectbox(label="Select Numerical or Categorical ", options=[None, 'Numerical', 'Categorical'])
        column_options = numeric_columns if variable_type == 'Numerical' else categorical_columns
        unianalysis_feature = st.sidebar.selectbox('Select Feature  ', [None] + list(column_options))

        if unianalysis_feature is not None:
            graph_list = ['Box Plot', 'Grouped Box Plot', 'Histogram'] if variable_type == 'Numerical' else ['Count Plot', 'Pie Chart']
            graph_type = st.sidebar.selectbox('Select Graph Type', [None] + graph_list)
            color = st.sidebar.color_picker("Select Color for the Graph", "#3498db")

            try:
                univariant_graph(unianalysis_feature, graph_type, color)
            except Exception :
                st.subheader(f'Change the Graph {graph_type} some other one')
                st.write("Cannot Perform the Plot Due to Overlength")

#Bivariant analysis 
    elif analysis_type == 'Bivariate':
        dynamic_content.empty()

        variable_type = st.sidebar.selectbox(label="Select Numerical or Categorical ", options=[None, 'Numerical', 'Categorical'])
        column_options = numeric_columns if variable_type == 'Numerical' else categorical_columns

        x_values=list(column_options)
        y_values=list(column_options)

        x_variable=st.sidebar.radio('Select X Axis Column for Analysis ',[None]+x_values)
        if x_variable is not None:
            y_values.remove(x_variable)

        y_variable=st.sidebar.radio('Select X Axis Column for Analysis ',[None]+y_values,key="x_axis_column_selection")

        if y_variable is not None :
            graph_list = ['Scatter Plot', 'Pair Plot', 'Bar Plot'] if variable_type == 'Numerical' else ['Count Plot']
            graph_type = st.sidebar.selectbox('Select Graph Type', [None] + graph_list)
            color = st.sidebar.color_picker("Select Color for the Graph", "#3498db")

            try:
                binvariant_graph(x_variable,y_variable,graph_type,color,column_options)
                pass
            except Exception :
                st.subheader(f'Change the Graph {graph_type} some other one')
                st.write("Cannot Perform the Plot Due to Overlength")

#Multivariate analysis 
    elif analysis_type == 'Multivariate':
        dynamic_content.empty()

        variable_type = st.sidebar.selectbox(label="Select Numerical or Categorical ", options=[None, 'Numerical', 'Categorical'])
        column_options = numeric_columns if variable_type == 'Numerical' else categorical_columns

        x_values=list(column_options)
        y_values=list(column_options)
        z_values=list(categorical_columns)

        x_variable=st.sidebar.radio('Select X Axis Column for Analysis ',[None]+x_values)
        if x_variable is not None:
            y_values.remove(x_variable)

        y_variable=st.sidebar.radio('Select Y Axis Column for Analysis ',[None]+y_values,key="x_axis_column_selection")
        if y_variable is not None :
            pass
        z_variable=st.sidebar.radio(f"Select Hue For {x_variable} and {y_variable}",[None]+z_values,key="y_axis_column_selection")

        if(z_variable is not None):

            graph_list = ['Scatter Plot', 'Pair Plot', 'Heat Map'] if variable_type == 'Numerical' else ['']
            graph_type = st.sidebar.selectbox('Select Graph Type', [None] + graph_list)
            
        elif(x_variable,y_variable,z_variable is None and column_options == 'Numerical'):

            graph_list = ['Heat Map'] if variable_type == 'Numerical' else ['']
            graph_type = st.sidebar.selectbox('Select Graph Type', [None] + graph_list)
            color = st.sidebar.color_picker("Select Color for the Graph", "#3498db")
            

            try:
                multivariant_graph(x_variable,y_variable,z_variable,graph_type,color,column_options)
                pass
            except Exception :
                st.subheader(f'Change the Graph {graph_type} some other one')
                st.write("Cannot Perform the Plot Due to Overlength")

    else:
        pass



if __name__ == "__main__":
    main()
