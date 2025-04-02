import time
import logging
import torch

# Importing Customized Classes.
from preprocess import Preprocess as prp
from quadtree import InitialQuadtree as quad
from visualise import Visualise as vis

# Configure logging to display time, level, and message
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if torch.cuda.is_available():
    logging.info("CUDA is available. PyTorch can use GPU for computations.")
else:
    logging.info("CUDA is not available. PyTorch will use CPU for computation.")

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
    start_date = '2015-01-01'
    end_date = '2020-12-31'
    df = prp.get_sample_data(df, start_date, end_date)

    if df.empty:
        raise ValueError("DataFrame is empty after after getting sample data choose past years.!")

    df = prp.add_daily_crime_count(df)

    df = prp.create_date_features(df)

    # Step 6: Data Split and Added Prediction Column with Zero Value.
    train_df, val_df = prp.train_test_df_split(df, train_size=0.8)
    train_df = quad.set_pred_zero(train_df)
    # val_df = quad.set_pred_zero(val_df)

    train_df['Crime_count'] = train_df['Crime_count'].fillna(0)
    val_df['Crime_count'] = val_df['Crime_count'].fillna(0)

    print(f'Training Dataset: \n {train_df}')

    print(f'Validation Dataset: \n {val_df}')

    print(train_df.columns)


############################# Create Quadtree #############################

    print(f"train_df Crime_count summary: {train_df['Crime_count'].describe()}")
    print(f"train_df Crime_count NaN count: {train_df['Crime_count'].isna().sum()}")

    # Step 7: Create Adaptive Quadtree.
    quadtree = quad.init_quadtree(train_df)

    # Step 10: Visualise the quadtree
    vis.visualize_quadtree(quadtree)

    # # After quadtree creation
    # quadtree.compute_stats()
    # logging.info("Computed node statistics.")

    # Time Count End.
    elapsed_time = time.time() - start_time
    logging.info(f"Script completed in {elapsed_time:.2f} seconds.")


