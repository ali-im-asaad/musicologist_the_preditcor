import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
from io import BytesIO
import base64

# Function to generate a download link for textual data
def download_link(object_to_download, download_filename, download_link_text):
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
    b64 = base64.b64encode(object_to_download.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'
    return href

# Function to generate a download link for plots
def get_image_download_link(fig, filename='plot.png', link_text='Download Plot'):
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# Set up the title and sidebar description
st.title('Music Valence Prediction Model')
st.sidebar.header('User Inputs & Actions')

# Sidebar - File uploader widget
uploaded_file = st.sidebar.file_uploader("1. Choose a CSV file for analysis", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, index_col=0)
    df = df.fillna(df.mean())

    st.sidebar.subheader("2. Select the Target Variable")
    target_variable = st.sidebar.selectbox('This variable will be predicted by the model.', df.columns)

    if st.sidebar.button('Show Data Description'):
        st.subheader('Data Description')
        st.write(df.describe())
        st.markdown(download_link(df.describe(), 'data_description.csv', 'Download Data Description'), unsafe_allow_html=True)

    if st.sidebar.button('Show Correlation Heatmap'):
        st.subheader('Correlation Heatmap')
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), ax=ax, annot=True, cmap='viridis')
        st.pyplot(fig)
        st.markdown(get_image_download_link(fig, 'correlation_heatmap.png', 'Download Heatmap'), unsafe_allow_html=True)

    target = df[target_variable]
    predictors = df.drop(columns=[target_variable])
    predictors_train, predictors_test, target_train, target_test = train_test_split(predictors, target, test_size=0.30, random_state=0)

    if st.sidebar.button('Train Model'):
        st.subheader('Model Training & Evaluation')
        linear_regressor = LinearRegression()
        sfs_forward = SequentialFeatureSelector(linear_regressor, n_features_to_select=3, direction="forward").fit(predictors_train, target_train)
        
        selected_features = sfs_forward.get_feature_names_out()
        st.write(f"Selected Features: {selected_features}")
        
        predictors_selected_train = predictors_train.loc[:, selected_features]
        predictors_selected_test = predictors_test.loc[:, selected_features]
        linear_regressor.fit(predictors_selected_train, target_train)
        
        r2_train = linear_regressor.score(predictors_selected_train, target_train)
        r2_test = linear_regressor.score(predictors_selected_test, target_test)
        st.write(f"R² (Training): {r2_train:.2f}, R² (Testing): {r2_test:.2f}")
        
        
        target_pred_train = linear_regressor.predict(predictors_selected_train)
        target_pred_test = linear_regressor.predict(predictors_selected_test)
        fig, ax = plt.subplots()
        ax.scatter(target_train, target_pred_train, label="Training Data")
        ax.scatter(target_test, target_pred_test, label="Testing Data")
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Actual vs. Predicted')
        ax.legend()
        st.pyplot(fig)

        # After making predictions
        r2_train_manual = r2_score(target_train, target_pred_train)
        r2_test_manual = r2_score(target_test, target_pred_test)

        st.write(f"Manual R² (Training): {r2_train_manual:.2f}, Manual R² (Testing): {r2_test_manual:.2f}")

        st.markdown(get_image_download_link(fig, 'actual_vs_predicted.png', 'Download Plot'), unsafe_allow_html=True)
