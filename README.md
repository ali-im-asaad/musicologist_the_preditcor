# Music Valence Prediction Model

## Overview
Al-Mutannabi: The Musicologist's Predictor is a Streamlit-based application designed to dynamically predict variables based on various attributes. It allows musicologists to upload Datasets as CSV/XLSX, dynamically and easily select a target variable, and select columns to exclude from predictors, then train a model to predict specific features tailored to the research use case. The app provides data descriptions, correlation heatmaps, and actual vs. predicted plots to analyze the predictions.

## Features
- Upload CSV data for analysis.
- Select target variable for prediction.
- Data description and download.
- Correlation heatmap visualization.
- Train linear regression model with feature selection.
- Actual vs. predicted comparison plot.

## Installation
To install and run this application, follow these steps:

1. Clone the repository or download the application files to your local machine.
2. Ensure you have Python 3.6 or later installed.
3. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Run the application:
   ```
   streamlit run app.py
   ```

## Usage
After launching the app, follow the instructions on the sidebar to upload your data, select the target variable, and interact with the model and visualizations.

## License
This project is proprietary and confidential. All rights reserved. Usage, modification, and distribution are allowed only with explicit written permission from the author. For inquiries regarding licensing or any other matters, please contact the author at ali.im.asaad@gmail.com

## Contributing
Contributions to improve the app are welcome. Please follow the standard fork and pull request workflow.

---

For more information, please refer to the application documentation or contact the repository owner.
