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
st.title("Al-Mutanabbi: The Musicologist's Predictor")
st.sidebar.header('Inputs & Actions')

# Sidebar - File uploader widget, now accepts CSV and Excel formats
uploaded_file = st.sidebar.file_uploader("1. Choose a CSV or Excel file for analysis", type=["csv", "xlsx"])
if uploaded_file is not None:
    # Check file extension to use the appropriate Pandas function
    file_extension = uploaded_file.name.split('.')[-1]
    if file_extension.lower() == 'csv':
        df = pd.read_csv(uploaded_file, index_col=0)
    elif file_extension.lower() == 'xlsx':
        df = pd.read_excel(uploaded_file, index_col=0)
    else:
        st.error("Unsupported file format! Please upload a CSV or Excel file.")
        st.stop()

    df = df.fillna(df.mean())

    # Let the researcher decide which columns to exclude
    columns_to_exclude = st.sidebar.multiselect('Select columns to exclude from predictors:', df.columns)
    # Ensure target variable selection is from columns not excluded
    potential_targets = [col for col in df.columns if col not in columns_to_exclude]
    st.sidebar.subheader("Select the Target Variable")
    target_variable = st.sidebar.selectbox('This variable will be predicted by the model.', potential_targets)
    
    if target_variable:
        # Set the dependent variable and predictors
        target = df[target_variable]
        predictors = df.drop(columns=[target_variable] + columns_to_exclude)
        
    # st.sidebar.subheader("2. Select the Target Variable")
    # target_variable = st.sidebar.selectbox('This variable will be predicted by the model.', df.columns)
    print(target_variable)
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

    # target = df[target_variable]
    # predictors = df.drop(columns=[target_variable])

    # num_columns_before = 3  # Adjust this number as needed based on your dataset
    # predictors = df.iloc[:, num_columns_before:]
    # predictors_train, predictors_test, target_train, target_test = train_test_split(predictors, target, test_size=0.30, random_state=0)

    randnumber = 0
    predictors_train, predictors_test, target_train, target_test = train_test_split(predictors, target, test_size=0.30, random_state=randnumber)




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

        # Function to interpret R² values
        def interpret_r2(r2):
            if r2 >= 0.75:
                return "Excellent model performance. The model explains a large portion of the variability in the response variable."
            elif r2 >= 0.5:
                return "Good model performance. The model explains a significant portion of the variability in the response variable, though there's room for improvement."
            elif r2 >= 0.25:
                return "Moderate model performance. The model explains some variability in the response variable, but it might not be sufficient for all purposes."
            else:
                return "Poor model performance. The model does not explain much of the variability in the response variable. Consider reviewing the model's assumptions, adding more features, or using a different model."

        # Display R² values and their interpretations
        st.write(f"R² (Training): {r2_train:.2f}, R² (Testing): {r2_test:.2f}")
        st.write(f"Interpretation (Training): {interpret_r2(r2_train)}")
        st.write(f"Interpretation (Testing): {interpret_r2(r2_test)}")
