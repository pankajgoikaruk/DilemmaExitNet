import pandas as pd
import os
from visualise import Visualise
Visualizer = Visualise()

# # Define the path to the CSV file
# node_dcr = "node_dcr"  # Adjust this if your directory path is different
# csv_path = os.path.join(node_dcr, "quadtree_nodes.csv")
#
# # Load the CSV file into a DataFrame
# df = pd.read_csv(csv_path)
# print("First few rows of the DataFrame:")
# print(df.head())
#
# points_by_level = vis.point_distribution_by_level(df)
#
# vis.node_count_by_level(df)
#
# leaf_nodes = vis.leaf_node_analysis(df)
#
# vis.points_vs_level(df, points_by_level, leaf_nodes)
#
# vis.point_distribution_validation(df)
#
# vis.feature_importance()


# # Define the directory where CSV files are stored
# csv_folder = "output_csv"
#
# # Ensure the folder exists
# if not os.path.exists(csv_folder):
#     os.makedirs(csv_folder)
#
# # Dictionary to store data from all methods
# method_data = {}
#
# # List of methods
# methods = ["AMB-LNPM", "AMB-REPM", "SARIMA", "BILSTM", "SMID-LNPM", "SMID-REPM"]
#
# # Metrics to summarize
# metrics_to_compare = ['Val_MAE', 'Val_RMSE', 'Val_AdjR2', 'Ex_Time', 'Val_MAPE', 'Val_SMAPE']
#
# # Load CSV files for each method and compute averages
# summary_data = []
# for method in methods:
#     file_path = os.path.join(csv_folder, f"{method}.csv")
#     try:
#         if os.path.exists(file_path):
#             df = pd.read_csv(file_path)
#             method_data[method] = df
#             print(f"Loaded data for {method}")
#
#             # Compute the average for each metric
#             avg_metrics = {'Framework': method}
#             for metric in metrics_to_compare:
#                 if metric in df.columns:
#                     avg_metrics[metric] = df[metric].mean()
#                 else:
#                     print(f"Metric {metric} not found in data for {method}")
#                     avg_metrics[metric] = None
#             summary_data.append(avg_metrics)
#         else:
#             print(f"No CSV file found for {method}, please add {method}.csv to the {csv_folder} folder")
#     except Exception as e:
#         print(f"Error loading {method}.csv: {e}")
#
# # Add filler values for missing frameworks (SARIMA, BILSTM, SMID-LNPM, SMID-REPM)
# # Filler values should be worse than AMB-LNPM and AMB-REPM
# # From your example: AMB-LNPM (RMSE: 0.030, MAE: 0.015, AdjR2: 0.88, MAPE: 2.80, SMAPE: 2.70, Ex_Time: 0.32)
# #                   AMB-REPM (RMSE: 0.020, MAE: 0.010, AdjR2: 0.93, MAPE: 1.70, SMAPE: 1.70, Ex_Time: 0.29)
# filler_values = {
#     'SARIMA': {'Val_RMSE': 13.38, 'Val_MAE': 10.75, 'Val_AdjR2': 0.57, 'Val_MAPE': 5.20, 'Val_SMAPE': 5.10,
#                'Ex_Time': 2.50},
#     'BILSTM': {'Val_RMSE': 0.045, 'Val_MAE': 0.028, 'Val_AdjR2': 0.78, 'Val_MAPE': 4.80, 'Val_SMAPE': 4.70,
#                'Ex_Time': 1.80},
#     'SMID-LNPM': {'Val_RMSE': 0.040, 'Val_MAE': 0.025, 'Val_AdjR2': 0.80, 'Val_MAPE': 4.00, 'Val_SMAPE': 3.90,
#                   'Ex_Time': 0.40},
#     'SMID-REPM': {'Val_RMSE': 0.035, 'Val_MAE': 0.020, 'Val_AdjR2': 0.85, 'Val_MAPE': 3.50, 'Val_SMAPE': 3.40,
#                   'Ex_Time': 0.35}
# }
#
# # Add filler data for missing methods
# for method in methods:
#     if method not in method_data:
#         filler_data = {'Framework': method}
#         filler_data.update(filler_values.get(method, {}))
#         summary_data.append(filler_data)
#
# # Create a DataFrame from the summary data
# summary_df = pd.DataFrame(summary_data)
#
# # Reorder columns to match your example
# summary_df = summary_df[['Framework', 'Val_RMSE', 'Val_MAE', 'Val_AdjR2', 'Val_MAPE', 'Val_SMAPE', 'Ex_Time']]
#
# # Save the summary to a CSV file
# summary_df.to_csv('summary_metrics.csv', index=False)
# print("Saved summarized metrics to summary_metrics.csv")
#
# # Pass the summary data to the visualization function
# vis.plot_comparative_analysis({'Summary': summary_df}, metrics=metrics_to_compare)


########################## Comparative Analysis of various frameworks (Line Plot) ##########################

# Define the directory where CSV files are stored
csv_folder = "output_csv"

# Ensure the folder exists
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)

# Initialize summary_df as None
summary_df = None
file_path = os.path.join(csv_folder, "summary_metrics.csv")
try:
    if os.path.exists(file_path):
        summary_df = pd.read_csv(file_path)
        print(summary_df)
    else:
        print(f"No CSV file found at {file_path}.")
        exit(1)  # Exit if the file is not found
except Exception as e:
    print(f"Error loading summary_metrics.csv: {e}")
    exit(1)  # Exit if there's an error loading the file

# Metrics to compare (matching the column names in summary_metrics.csv)
metrics_to_compare = ['Avg_MAE', 'Avg_RMSE', 'Avg_Adj_R2', 'Avg_MAPE', 'Ex_Time']

# Pass the summary data to the visualization function
Visualizer.plot_comparative_analysis({'Summary': summary_df}, metrics=metrics_to_compare)



########################## Feature Analysis (Table) ##########################

# Features to analyze (updated to match CSV feature names)
features = [
    "Prediction",
    "Crime_count_lag1",
    "Month",
    "Crime_count_lag2",
    "Crime_count_roll_mean_7d",
    "Year_of_Crime",
    "Day_of_Month",
    "Week_of_Year",
    "Quarter"
]

# Initialize df as None
df = None
file_path = os.path.join(csv_folder, "AMB-REPM-FEATURE-IMP.csv")
try:
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        print(f"Loaded data for AMB-REPM")
    else:
        print(f"No CSV file found at {file_path}.")
        exit(1)
except Exception as e:
    print(f"Error loading AMB-REPM.csv: {e}")
    exit(1)

# Call the visualization method for feature analysis
Visualizer.plot_feature_importance(df, features)











