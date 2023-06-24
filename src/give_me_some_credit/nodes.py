import pandas as pd
import matplotlib.pyplot as plt
import shap
import nannyml as nml
import missingno as msno
import seaborn as sns
import os
import lightgbm as lgb
import numpy as np
import mlflow

from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Tuple
from sklearn.impute import KNNImputer
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif

def read_data(file_path: str) -> pd.DataFrame:
    """
    Read CSV file and return a pandas DataFrame object.
    
    :param file_path: str, path to the CSV file
    :return: pd.DataFrame, pandas dataframe with the data from the input file
    """
    return pd.read_csv(file_path, encoding='utf-8')

def plot_boxplot(data: pd.DataFrame, output_dir: str) -> None:
    """
    Plot and save the boxplot of each variable in the input data.

    :param data: pd.DataFrame, input data
    :param output_dir: str, output directory to save the plots
    """
    for col in data.columns[2:]:
        if data[col].dtype in ['int64', 'float64']:
            sns.boxplot(x=data[col])
            plt.title(f'Boxplot of {col}')
            plt.savefig(os.path.join(output_dir, f"boxplot_{col}.png"))
            plt.close()

def plot_null_values(data: pd.DataFrame, output_dir: str) -> None:
    """
    Plot and save the null values present in the input data.

    :param data: pd.DataFrame, input data
    :param output_dir: str, output directory to save the plot
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    msno.matrix(data.iloc[:, 2:], figsize=(14, 6), fontsize=12, sparkline=False, labels=True, color=(0.2, 0.3, 0.5))
    plt.title('Null Values Present in Data', fontsize=20)
    plt.xticks(rotation=90, fontsize=12)
    plt.xlabel('Columns', fontsize=14)
    plt.ylabel('Row Index', fontsize=14)
    
    plt.savefig(os.path.join(output_dir, "null_values_matrix.png"), bbox_inches='tight')
    plt.close()

def generate_statistics(data: pd.DataFrame) -> Dict:
    """
    Generate descriptive statistics for the input data.
    
    :param data: pd.DataFrame, input data
    :return: dict, dictionary containing descriptive statistics
    """
    statistics = data.describe(include='all').to_dict()
    return statistics

def preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input data. Steps include handling missing values, eliminating outliers, and scaling features.
    
    :param data: pd.DataFrame, input data in raw format
    :return: pd.DataFrame, preprocessed data
    """
    # Removing the First Column since it represents the customer index 
    data = data.iloc[:, 1:]

    # Handling Missing Values 
    # Preparing the MonthlyIncome data for kNN imputation
    monthly_income_data = data['MonthlyIncome'].values.reshape(-1, 1)

    # Imputing the MonthlyIncome variable with kNN
    imputer = KNNImputer(n_neighbors=5, weights='uniform')
    monthly_income_imputed = imputer.fit_transform(monthly_income_data)

    # Replacing the original MonthlyIncome column with the imputed data
    data['MonthlyIncome'] = monthly_income_imputed

    # Filling with the mode for the categorical feature
    mode_dependents = data['NumberOfDependents'].mode()[0]
    data['NumberOfDependents'].fillna(mode_dependents, inplace=True)

    # Custom outlier removal
    data=data[data['age']>=21]
    data = data[data['DebtRatio'] <= 210000]
    data = data[data['MonthlyIncome'] <= 2000000]
    data = data[data['NumberOfTime30-59DaysPastDueNotWorse'] <= 80]
    data = data[data['NumberOfTime60-89DaysPastDueNotWorse'] <= 80]
    data = data[data['NumberOfTimes90DaysLate'] <= 80]
    data = data[data['RevolvingUtilizationOfUnsecuredLines'] <= 40000]

    # #  # Scaling Features (Optional) 
    # # scaler = StandardScaler()
    # # numerical_features = ['RevolvingUtilizationOfUnsecuredLines', 'age',
    # #                       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
    # #                       'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
    # #                       'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
    # #                       'NumberOfTime60-89DaysPastDueNotWorse']
                          
    # data[numerical_features] = scaler.fit_transform(data[numerical_features])

    return data

def plot_correlation_matrix(data: pd.DataFrame, output_dir: str) -> None:
    """
    Plot and save the correlation matrix for the input preprocessed data.

    :param data: pd.DataFrame, preprocessed input data
    :param output_dir: str, output directory to save the plot
    """
    # Dropping the target variable if present, to focus on the correlation matrix only for features
    if 'SeriousDlqin2yrs' in data.columns:
        data = data.drop('SeriousDlqin2yrs', axis=1)

    # Calculating the correlation matrix
    correlation_matrix = data.corr()

    # Creating a heatmap to visualize the correlation matrix
    plt.figure(figsize=(20, 12)) 
    sns.set(font_scale=0.8) # Set a smaller font scale for better readability
    ax = sns.heatmap(correlation_matrix,annot=True, cmap='YlGnBu', linewidths=0.5)
    ax.set_title('Correlation Matrix of Preprocessed Data', fontsize=20)

    # Saving the heatmap plot to a file with higher DPI and better resolution
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"), dpi=300)
    plt.close()

def data_drift(data_reference: pd.DataFrame, data_analysis: pd.DataFrame):   
    """
    Calculate data drift on input data.

    :param data_reference: pd.DataFrame, reference data
    :param data_analysis: pd.DataFrame, data to analyze against reference data
    :return: pd.DataFrame, data drift results
    """
    
    # Extracting column names from the preprocessed_data
    column_names = data_reference.columns.tolist()

    # Identifying categorical and numerical columns
    categorical_columns = list(data_reference.select_dtypes(include=['object', 'category']).columns)
    numerical_columns = [col for col in column_names if col not in categorical_columns]

    # Defining the threshold for the test as parameters in the parameters catalog
    constant_threshold = nml.thresholds.ConstantThreshold(lower=0.3, upper=0.7)
    constant_threshold.thresholds(data_reference)

    # Initializing the object that will perform the Univariate Drift calculations
    univariate_calculator = nml.UnivariateDriftCalculator(
        column_names=column_names,
        treat_as_categorical=categorical_columns,
        chunk_size=50,
        categorical_methods=['jensen_shannon'],thresholds={"jensen_shannon": constant_threshold}) 

    univariate_calculator.fit(data_reference)
    results = univariate_calculator.calculate(data_analysis).to_df()

    # Generating a report for some numeric features using KS test and Evidently AI
    data_drift_report = Report(metrics=[DataDriftPreset(stattest_threshold=0.05)])

    # Running the report for numerical columns
    data_drift_report.run(current_data=data_analysis[numerical_columns], reference_data=data_reference[numerical_columns], column_mapping=None)
    data_drift_report.save_html("data/08_reporting/data_drift_report.html")
    return results


def train_test_data_split(data: pd.DataFrame, test_size: float, random_state: int) -> tuple:
    """
    Split the input data into training and testing sets.
    
    :param data: pd.DataFrame, input data
    :param test_size: float, proportion of the data to be used as test data:param random_state: int, seed used by random number generator
    :return: tuple, a tuple containing training, testing, and target dataframes
    """
    X = data.drop('SeriousDlqin2yrs', axis=1)  
    y = data['SeriousDlqin2yrs']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # Converting Series objects to DataFrames
    y_train = y_train.to_frame()
    y_test = y_test.to_frame()
    
    return X_train, X_test, y_train, y_test

def plot_learning_curve(estimator, title, X, y, cv, output_path):
    """
    This function plots the learning curve for a given model using the learning curve function from
    sklearn.

    :param estimator: the model used for training and validation
    :param title: title for the plot
    :param X: pd.DataFrame, training data
    :param y: pd.DataFrame, training labels
    :param cv: int, cross-validation value
    :param output_path: str, path to save the learning curve plot
    """

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv)

    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")

    # Save the learning curve plot to a file
    plt.savefig(output_path)

def model_train(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame,
                shap_output_path: str, overfitting_output_path: str, cv: int, 
                feature_selection: bool = False) -> Tuple[RandomForestClassifier, Optional[pd.Index]]:
    """
    Train a RandomForestClassifier or LightGBM model using GridSearchCV for
    hyperparameter tuning and cross-validation, and compare both models based on their mean 
    cross-validated score.
    
    :param X_train: pd.DataFrame, training data
    :param X_test: pd.DataFrame, testing data
    :param y_train: pd.DataFrame, target training data
    :param shap_output_path: str, path to save the SHAP feature importance plot
    :param cv: int, the number of cross-validation folds :param feature_selection: bool, whether to perform feature selection (default is False)
    :return: Tuple, containing the trained classifier model (RandomForestClassifier or LightGBMClassifier) and selected features (`pd.Index` or `None`)
    """

    selected_features = None

    if feature_selection:
        # Storing the original feature names
        original_columns = X_train.columns

        # Performing feature selection using one of the methods.
        kbest_selector = SelectKBest(score_func=mutual_info_classif, k=8)
        X_train = kbest_selector.fit_transform(X_train, y_train)
        X_test = kbest_selector.transform(X_test)

        # Getting the selected feature indices
        selected_indices = kbest_selector.get_support(indices=True)

        # Getting the selected feature names
        selected_features = original_columns[selected_indices]

        # Updating the DataFrames with the selected features' names
        X_train = pd.DataFrame(X_train, columns=selected_features)
        X_test = pd.DataFrame(X_test, columns=selected_features)
    else:
        selected_features = X_train.columns

    # Instantiating the Random Forest Classifier and LightGBM Classifier
    rf = RandomForestClassifier()
    gb = lgb.LGBMClassifier()

    param_grid_rf = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 7]}
    param_grid_gb = {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.3], "max_depth": [3, 5, 7]}

    # Using GridSearchCV for hyperparameter tuning and cross-validation for Random Forest Classifier
    grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=cv)
    grid_search_rf.fit(X_train, y_train)

    # Using GridSearchCV for hyperparameter tuning and cross-validation forLightGBM Classifier
    grid_search_gb = GridSearchCV(estimator=gb, param_grid=param_grid_gb, cv=cv)
    grid_search_gb.fit(X_train, y_train)

    # Comparing mean cross-validated scores and select the best model
    if grid_search_rf.best_score_ > grid_search_gb.best_score_:
        best_model = grid_search_rf.best_estimator_
        shap_values = shap.TreeExplainer(best_model).shap_values(X_test)

         # Plotting the SHAP feature importance
        shap.summary_plot(shap_values[1], X_test, show=False)
          # Saving the SHAP plot to a PNG file
        plt.savefig(shap_output_path)
        plot_learning_curve(best_model, "Random Forest Learning Curve", X_train, y_train, cv, overfitting_output_path)
    else:
        best_model = grid_search_gb.best_estimator_
        shap_values = shap.TreeExplainer(best_model).shap_values(X_test)
        # Plotting the SHAP feature importance
        shap.summary_plot(shap_values[1], X_test, show=False)

        # Saving the SHAP plot to a PNG file
        plt.savefig(shap_output_path)
        plot_learning_curve(best_model, "LightGBM Learning Curve", X_train, y_train, cv, overfitting_output_path)


    return best_model, selected_features

def write_parameters_to_file(model, feature_selection: bool, output_filepath: str):
    """
    Write the parameters and the best model to a text file.

    :param model: Best model retrieved from model train
    :param feature_selection: bool, whether feature selection was used
    :param output_filepath: str, path to save the text file
    """

    with open(output_filepath, "w") as f:
        f.write(f"{'Model Type:'}\n{str(type(model))}\n\n")
        f.write(f"{'Model Parameters:'}\n{model.get_params()}\n\n")
        f.write(f"{'Feature Selection:'}\n{feature_selection}\n")

def test_model(model, X_test: pd.DataFrame, y_test: pd.DataFrame, selected_features: Optional[pd.Index]) -> dict:
    """
    Evaluate the given model using the test dataset and selected features.

    Parameters
    ----------
    model: {classifier object}
        Trained model object to be tested.
    X_test: pd.DataFrame
        Test dataset containing features used for testing the model.
    y_test: pd.DataFrame
        Test dataset containing target variable corresponding to the X_test dataset.
    selected_features: Optional[pd.Index]
        Optional list of selected feature names to be used for testing. If provided, the function will only
        use these features for testing. Default is None.

    Returns
    -------
    dict
        A dictionary containing evaluation metrics (accuracy, f1_score, classification_report) for the tested model.
    """

    if selected_features is not None:
        X_test = X_test[selected_features]
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    classificationReport = classification_report(y_test, y_pred, output_dict=True)
    
    return {"accuracy": accuracy, "f1_score": f1, "classification_report": classificationReport}


def log_mlflow_artifacts(mlflow_artifacts_directory: str):
    """
    This function logs the artifacts from a given directory to an MLflow run.

    Artifacts can be any file generated during a machine learning
    experiment, such as result files or models.

    The function uses the `mlflow.log_artifact()` method to log each
    file located in the given directory.

    Parameters:
    -----------
    mlflow_artifacts_directory (str): A string representing the path to
                                      the artifacts directory containing
                                      the files to be logged.

    Example:
    --------
    log_mlflow_artifacts("path/to/artifacts_directory")

    """
    # Getting the list of files in the artifacts directory
    artifact_files = os.listdir(mlflow_artifacts_directory)

    # Logging each file as an artifact
    for filename in artifact_files:
        file_path = os.path.join(mlflow_artifacts_directory, filename)
        mlflow.log_artifact(file_path)
