from kedro.pipeline import Pipeline, node
from .nodes import (
    read_data,
    preprocessing,
    generate_statistics,
    train_test_data_split,
    model_train,
    test_model,
    data_drift,
    plot_boxplot,
    plot_null_values,
    plot_correlation_matrix,
    write_parameters_to_file,
    log_mlflow_artifacts
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            # Read raw data
            node(read_data, "params:raw_data_path", "raw_data"),
            
            # Generate statistics for raw data
            node(generate_statistics, "raw_data", "statistics"),
        
            # Plot the boxplot of each variable in the raw data
            node(plot_boxplot, ["raw_data", "params:output_dir"], None),

            # Plot the null values present in the raw data
            node(plot_null_values, ["raw_data", "params:output_dir"], None),
            
            # Preprocess raw data
            node(preprocessing, "raw_data", "preprocessed_data"),

            # Plot the correlation matrix for preprocessed data
            node(plot_correlation_matrix, ["preprocessed_data", "params:output_dir"], None),
            
            # Analyze data drift
            node(func=data_drift,
                 inputs=["preprocessed_data", "preprocessed_data"],
                 outputs="drift_result",
                 name="drift_analysis"),
            
            # Split preprocessed data into train and test sets
            node(train_test_data_split, 
                 ["preprocessed_data", "params:test_size", "params:random_state"],
                 ["X_train", "X_test", "y_train", "y_test"]),
            
            # Train the model using train data and tune the hyperparameters
            node(
                model_train,
                ["X_train", "X_test", "y_train", "params:shap_output_path", "params:overfitting_output_path","params:cv", "params:feature_selection"],
                ["trained_model", "selected_features"],
                name="train_model_node",
                 ),
            node(write_parameters_to_file,
                ["trained_model", "params:feature_selection", "params:output_filepath"],
                None),
                
            # Test the trained model on test data and compute evaluation metrics
            node(
                test_model,
                ["trained_model", "X_test", "y_test", "selected_features"],
                "evaluation_metrics",
                name="test_model_node",
            ),
            node(
                func=log_mlflow_artifacts,
                inputs="params:mlflow_artifacts_directory",
                outputs=None,
                name="log_mlflow_artifacts",
            ),
        ]
    )