# import pandas as pd
# import torch
# import os
# import time
# import logging
#
# if torch.cuda.is_available():
#     print("CUDA is available. PyTorch can use GPU for Computations.")
# else:
#     print("CUDA is not available. PyTorch will use CPU for computational.")
#
# if __name__ == "__main__":
#
#     logging.info("Script started.")
#
#     # Optional: Record the start time
#     start_time = time.time()
#
#     path = "C:/Users/wwwsa/PycharmProjects/Chicago_Crime_Data/Chicago_Crime_2001_to_2024.csv"
#
#     df = pd.read_csv(path)
#
#     print(df)
#
#
#

#######################################################################################################################

# import os
# import time
# import logging
# import pandas as pd
# import torch
#
# # Configure logging to display time, level, and message
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#
# if torch.cuda.is_available():
#     logging.info("CUDA is available. PyTorch can use GPU for computations.")
# else:
#     logging.info("CUDA is not available. PyTorch will use CPU for computation.")
#
#
# def load_and_preprocess_data(path, required_columns=None):
#     """
#     Loads data from a CSV file and performs basic preprocessing.
#
#     Steps:
#     - Check if the file exists.
#     - Load the CSV file.
#     - Log the shape of the data.
#     - Drop rows with missing values in the specified required columns.
#     - Convert date columns (if any) to datetime.
#
#     Parameters:
#         path (str): Path to the CSV file.
#         required_columns (list or None): List of columns that must not have missing values.
#             If None, all columns are considered.
#     """
#     if not os.path.exists(path):
#         logging.error(f"File not found: {path}")
#         return None
#
#     try:
#         df = pd.read_csv(path)
#         logging.info(f"Data loaded successfully from {path}.")
#         logging.info(f"Initial data shape: {df.shape}")
#
#         if required_columns is not None:
#             # Log how many missing values in required columns
#             missing_required = df[required_columns].isnull().sum().sum()
#             logging.info(f"Total missing values in required columns {required_columns}: {missing_required}")
#             if missing_required > 0:
#                 df = df.dropna(subset=required_columns)
#                 logging.info(f"After dropping rows with missing required values, data shape: {df.shape}")
#
#         # # Convert potential date columns to datetime (if the column name contains 'date')
#         # date_columns = [col for col in df.columns if 'date' in col.lower()]
#         # for col in date_columns:
#         #     try:
#         #         df[col] = pd.to_datetime(df[col])
#         #         logging.info(f"Converted column '{col}' to datetime.")
#         #     except Exception as e:
#         #         logging.warning(f"Could not convert column '{col}' to datetime: {e}")
#         #
#         # return df
#
#     except Exception as e:
#         logging.error(f"Error loading data: {e}")
#         return None
#
#
# if __name__ == "__main__":
#     logging.info("Script started.")
#     start_time = time.time()
#
#     # Define the path to your CSV file
#     path = "C:/Users/wwwsa/PycharmProjects/Chicago_Crime_Data/Chicago_Crime_2001_to_2024.csv"
#
#     # Specify the required columns for your project
#     required_columns = ['Date', 'Longitude', 'Latitude'] # 'Time',
#
#     # Load and preprocess the data using the required columns check
#     df = load_and_preprocess_data(path, required_columns=required_columns)
#
#     if df is not None:
#         # Print the first few rows for verification
#         print(df.head())
#
#     elapsed_time = time.time() - start_time
#     logging.info(f"Script completed in {elapsed_time:.2f} seconds.")

#######################################################################################################################


import os
import time
import logging
import pandas as pd
import torch

# Configure logging to display time, level, and message
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if torch.cuda.is_available():
    logging.info("CUDA is available. PyTorch can use GPU for computations.")
else:
    logging.info("CUDA is not available. PyTorch will use CPU for computation.")


def standardize_column_names(df):
    """
    Standardizes column names for date, time, longitude, and latitude.
    If the data contains a combined date-time column (e.g., "datetime") and no separate "Time"
    column, it attempts to split it into separate "Date" and "Time" columns.
    """
    # Mapping of standard column names to possible synonyms (using lowercase for matching)
    column_aliases = {
        "Date": ["date", "incident_date", "occurrence_date", "reported_date", "datetime", "date_time", "timestamp"],
        "Time": ["time", "incident_time", "reported_time"],
        "Longitude": ["longitude", "long", "lng", "x_coordinate", "x"],
        "Latitude": ["latitude", "lat", "y_coordinate", "y"]
    }

    renamed_columns = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        for standard, aliases in column_aliases.items():
            if col_lower in [alias.lower() for alias in aliases]:
                renamed_columns[col] = standard
                break  # Once a match is found, no need to check other aliases.

    if renamed_columns:
        logging.info(f"Renaming columns based on predefined mapping: {renamed_columns}")
    df = df.rename(columns=renamed_columns)

    ########################## Logic 1 ##########################

    # Check for any required standard columns that are missing.
    required_columns = ["Date", "Time", "Longitude", "Latitude"]
    for standard in required_columns:
        if standard not in df.columns:
            # Special handling: if 'Time' is missing but 'Date' exists, check if 'Date' might have time info.
            if standard == "Time" and "Date" in df.columns:
                sample = df["Date"].dropna().iloc[0]
                try:
                    parsed = pd.to_datetime(sample, errors='coerce')
                    # If the parsed time is not exactly midnight, assume the column has time info.
                    if parsed is not None and parsed.time() != pd.Timestamp("00:00:00").time():
                        # Automatically split the Date column.
                        df["Date"] = pd.to_datetime(df["Date"])
                        df["Time"] = df["Date"].dt.strftime("%H:%M:%S")
                        df["Date"] = df["Date"].dt.date
                        logging.info(
                            "Detected combined 'Date' column with time information. Split into 'Date' and 'Time'.")
                        continue  # No need to prompt for 'Time'.
                except Exception as e:
                    logging.warning(f"Could not parse the 'Date' column to detect time information: {e}")

            # If the column is still missing, prompt the user.
            print(f"\nRequired column '{standard}' is not found in the dataset.")
            print(f"Available columns: {list(df.columns)}")
            user_input = input(
                f"Please enter the column name from the dataset that corresponds to '{standard}': ").strip()
            if user_input in df.columns:
                df = df.rename(columns={user_input: standard})
                logging.info(f"Renamed column '{user_input}' to '{standard}' based on user input.")
            else:
                logging.warning(
                    f"User provided column name '{user_input}' not found in the dataset. '{standard}' remains missing.")

    # Final check: if 'Date' exists but 'Time' is still missing, try to split if possible.
    if "Date" in df.columns and "Time" not in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"])
            df["Time"] = df["Date"].dt.strftime("%H:%M:%S")
            df["Date"] = df["Date"].dt.date
            logging.info("Split combined 'Date' column into separate 'Date' and 'Time' columns (final check).")
        except Exception as e:
            logging.warning(f"Failed to split combined 'Date' column during final check: {e}")

    return df


def load_and_preprocess_data(path, required_columns):
    """
    Loads data from a CSV file and performs preprocessing:
      - Checks for file existence.
      - Loads the CSV.
      - Standardizes column names using known synonyms.
      - Extracts only the required columns.
      - Drops rows with missing values in the required columns.

    Parameters:
        path (str): Path to the CSV file.
        required_columns (list): List of standardized column names needed.

    Returns:
        df (DataFrame): Preprocessed DataFrame or None if errors occur.
    """
    if not os.path.exists(path):
        logging.error(f"File not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
        logging.info(f"Data loaded successfully from {path}.")
        logging.info(f"Initial data shape: {df.shape}")

        # Standardize column names to our required names.
        df = standardize_column_names(df)
        logging.info(f"Data columns after standardization: {list(df.columns)}")

        # Check if the required columns exist.
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            logging.error(f"Required columns missing after standardization: {missing_required}")
            return None

        # Extract only the required columns.
        df = df[required_columns]
        logging.info(f"Data shape after extracting required columns {required_columns}: {df.shape}")

        # Drop rows with missing values in the required columns.
        missing_count = df.isnull().sum().sum()
        logging.info(f"Missing values in required columns before drop: {missing_count}")
        if missing_count > 0:
            df = df.dropna(subset=required_columns)
            logging.info(f"Data shape after dropping rows with missing required values: {df.shape}")

        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None


if __name__ == "__main__":
    logging.info("Script started.")
    start_time = time.time()

    # Define the path to your CSV file.
    # path = "C:/Users/wwwsa/PycharmProjects/Chicago_Crime_Data/Chicago_Crime_2001_to_2024.csv"
    path = "C:/Users/wwwsa/PycharmProjects/NYC_Crime_Data/USA_Crime_2008_to_2017.csv"

    # Specify the required standardized columns.
    required_columns = ['Date', 'Time', 'Longitude', 'Latitude']

    # Load and preprocess the data.
    df = load_and_preprocess_data(path, required_columns)

    if df is not None:
        # Display the first few rows for verification.
        print(df.head())

    elapsed_time = time.time() - start_time
    logging.info(f"Script completed in {elapsed_time:.2f} seconds.")


