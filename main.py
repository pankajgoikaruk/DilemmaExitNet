import os
import time
import logging
import pandas as pd
import torch
import datetime

# Configure logging to display time, level, and message
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if torch.cuda.is_available():
    logging.info("CUDA is available. PyTorch can use GPU for computations.")
else:
    logging.info("CUDA is not available. PyTorch will use CPU for computation.")

# Split datetime it into separate "Date" and "Time" columns.
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
    path = "C:/Users/wwwsa/PycharmProjects/Chicago_Crime_Data/Chicago_Crime_2001_to_2024.csv"
    # path = "C:/Users/wwwsa/PycharmProjects/NYC_Crime_Data/USA_Crime_2008_to_2017.csv"

    # Specify the required standardized columns.
    required_columns = ['Date', 'Time', 'Longitude', 'Latitude']

    # Load and preprocess the data.
    df = load_and_preprocess_data(path, required_columns)

    if df is not None:
        # Display the first few rows for verification.
        print(df.head())

        # Check if the 'Date' column is in a proper date/datatime format.
        sample_date = df["Date"].dropna().iloc[0]
        if isinstance(sample_date, (pd.Timestamp, datetime.date, datetime.datetime)):
            logging.info("'Date' column is in a valide date/time format.")
        else:
            logging.warning("'Date' column is not in a recognized date/time format. Attempting conversion.")
            try:
                df["Date"] = pd.to_datetime(df["Date"])
                logging.info("Conversion of 'Date' column to datetime format successful.")
            except Exception as e:
                logging.error(f"Conversion of 'Date' column failed: {e}")


    # Display the first few rows for verification.
    print(df.head())

    elapsed_time = time.time() - start_time
    logging.info(f"Script completed in {elapsed_time:.2f} seconds.")


