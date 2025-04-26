import time
import logging
import torch
import os

# Importing Customized Classes.
from preprocess import Preprocess as prp
from quadtree import InitialQuadtree
from visualise import Visualise as vis
import quadtree

# Create an instance
initquad_instance = InitialQuadtree()

# Configure logging to display time, level, and message
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if torch.cuda.is_available():
    logging.info("CUDA is available. PyTorch can use GPU for computations.")
else:
    logging.info("CUDA is not available. PyTorch will use CPU for computation.")

# Define directories.
node_dcr = 'node_dcr'

def setup_directories(dir_list):
    """
    Create directories if they do not exist.
    """
    for directory in dir_list:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created directory: {directory}")

# Set up the directories
setup_directories([node_dcr])

############################# MAIN SCRIPT ###########################

if __name__ == "__main__":
    logging.info("Script started.")
    # Time Count Start.
    start_time = time.time()

    # Define the path to your CSV file.
    # path = "C:/Users/wwwsa/PycharmProjects/Chicago_Crime_Data/Chicago_Crime_2001_to_2024.csv"
    path = "C:/Users/wwwsa/PycharmProjects/NYC_Crime_Data/USA_Crime_2008_to_2017.csv"

    # Specify the required standardized columns.
    required_columns = ['Date', 'Time', 'Longitude', 'Latitude']

    # Load and preprocess the data.
    df = prp.load_and_preprocess_data(path, required_columns)

    # # Print the data types to the console.
    # print("\nData types before verification:")

    # Verify data types for the required columns.
    if df is not None:
        df = prp.verify_data_types(df)
    else:
        print("Dataset is empty!")

    # Print the data types to the console.
    print("\nData types after verification:")

    # Sort the data by 'Date' in ascending order
    df = df.sort_values(by='Date', ascending=True)

    # Optional: Get sample data.
    start_date = '2016-12-01' # 2016-12-01
    end_date = '2017-12-31'
    df = prp.get_sample_data(df, start_date, end_date)

    if df.empty:
        raise ValueError("DataFrame is empty after after getting sample data choose past years.!")

    df = prp.add_daily_crime_count(df)

    df = prp.create_date_features(df)

    import matplotlib.pyplot as plt

    # # Assuming your original dataset has Longitude and Latitude
    # plt.scatter(df["Longitude"], df["Latitude"], s=1, alpha=0.5)
    # plt.title("Spatial Distribution of NYC Crime Data")
    # plt.xlabel("Longitude")
    # plt.ylabel("Latitude")
    # plt.show()

    print(f"Raw train_df Crime_count stats: {df['Crime_count'].describe().to_dict()}")

    # Step 6: Data Split and Added Prediction Column with Zero Value.
    train_df, val_df = prp.train_test_df_split(df, train_size=0.8)
    train_df = initquad_instance.set_pred_zero(train_df)
    val_df = initquad_instance.set_pred_zero(val_df)

    train_df['Crime_count'] = train_df['Crime_count'].fillna(0)
    val_df['Crime_count'] = val_df['Crime_count'].fillna(0)

    print(f'Training Dataset: \n {train_df}')

    print(f'Validation Dataset: \n {val_df}')

    print(train_df.columns)


############################# Create Quadtree #############################
    # Traverse the quadtree
    start_time = time.time()

    print(f"train_df Crime_count summary: {train_df['Crime_count'].describe()}")
    print(f"train_df Crime_count NaN count: {train_df['Crime_count'].isna().sum()}")

    constants = quadtree.calculate_constants(len(train_df))
    logging.info(f"Calculated constants: {constants}")

    # Step 7: Create Adaptive Quadtree.
    quadtree = initquad_instance.init_quadtree(train_df, constants, initquad_instance)

    # Perform merging of small leaf nodes
    quadtree.merge_small_leaf_nodes(threshold=5000)  # 1000

    quadtree.train_on_quadtree()  # Train on full train_df

    quadtree.evaluate_on_validation(val_df)  # Evaluate on val_df

    quadtree.traverse_quadtree()
    quadtree.evaluate_quadtree()
    print(f"Quadtree details saved to {node_dcr}/quadtree_nodes.csv")

    # Step 10: Visualise the quadtree
    vis.visualize_quadtree(quadtree)

    vis.structural_performance_metrics()

    vis.variance_of_points_in_leaf_nodes()

    vis.density_distribution()

    # # After quadtree creation
    # quadtree.compute_stats()
    # logging.info("Computed node statistics.")

    end_time = time.time()

    # Time Count End.
    elapsed_time = time.time() - start_time
    logging.info(f"Total Script completed in {elapsed_time:.2f} seconds.")
    # print(f"Script completed in {end_time - start_time:.2f} seconds.")


