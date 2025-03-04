import pandas as pd
import logging
import torch
import os
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar


class Preprocess:
    def __init__(self):
        pass

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

    # Load dataset.
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
            df = Preprocess.standardize_column_names(df)
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

    # Verify data type of date and coordinates and also create new temporal features.
    def verify_data_types(df):
        """
        Verifies that:
          - The 'Date' column is in datetime or date format.
          - The 'Longitude' and 'Latitude' columns are numeric.
          - The 'Time' column is a string (or can be treated as such).

        If the 'Date' column is not in the proper format, it converts it using pd.to_datetime.
        Similarly, it converts Longitude and Latitude to numeric types if needed.
        """
        logging.info("Verifying data types of required columns:")

        # Check the 'Date' column.
        if df["Date"].dtype == object:
            try:
                logging.info("Converting 'Date' column to Date Format.")
                df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
                logging.info("Successfully converted 'Date' column to date format.")
                # return df
            except Exception as e:
                logging.info("The 'Date' column is in a proper date/datetime format.")

        # Check the 'Time' column.
        if df["Time"].dtype == object:
            try:
                logging.info("Converting 'Time' column to Time Format.")
                df["Time"] = pd.to_datetime(df["Time"], format='%H:%M:%S', errors='coerce')#.dt.time
                logging.info("Successfully converted 'Time' column to Time format.")
                # Extract numeric time features.
                df["Hour"] = df["Time"].dt.hour
                df["Minute"] = df["Time"].dt.minute
                df["Second"] = df["Time"].dt.second

                # Convert 'Time' column to a time-only string (dropping the default date)
                df["Time"] = df["Time"].dt.strftime('%H:%M:%S')
            except Exception as e:
                logging.info("The 'Time' column is in a proper datetime format.")

        # Check the 'Longitude' and 'Latitude' columns.
        for coord in ["Longitude", "Latitude"]:
            if not pd.api.types.is_numeric_dtype(df[coord]):
                logging.warning(f"The '{coord}' column is not numeric. Attempting conversion...")
                df[coord] = pd.to_numeric(df[coord], errors='coerce')
                if pd.api.types.is_numeric_dtype(df[coord]):
                    logging.info(f"Successfully converted '{coord}' column to numeric type.")
                else:
                    logging.error(f"Failed to convert '{coord}' column to numeric type.")

        # Log the final data types
        logging.info("Final data types: \n")
        logging.info(f'\n{df.dtypes}')

        return df

    # Get sample data.
    @staticmethod
    def get_sample_data(df, start_date=None, end_date=None):
        # Filtering the DataFrame for the chosen date range
        sample_df = pd.DataFrame(df[(df['Date'] >= start_date) & (df['Date'] <= end_date)])
        return sample_df

    # Count daily crime and add to the df.
    @staticmethod
    def add_daily_crime_count(df):
        """
        Adds a 'crime_count' column that represents the number of crimes occurring on each date.

        - Groups the data by 'Date' and counts occurrences.
        - Merges the count back to the original DataFrame.
        """
        logging.info("Calculating daily crime count...")

        # Ensure 'Date' is in datetime format
        if df["Date"].dtype != "datetime64[ns]":
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Calculate daily crime count
        daily_counts = df.groupby("Date").size().reset_index(name="crime_count")

        # Merge with original DataFrame
        df = df.merge(daily_counts, on="Date", how="left")

        logging.info("Successfully added 'crime_count' column.")
        return df

    # Extracts multiple features from the 'Date' column to enhance model learning.
    def create_date_features(df):
        """
        Features Created:
          - Day_of_Week (0=Monday, 6=Sunday)
          - Is_Weekend (1 if Saturday/Sunday, else 0)
          - Day_of_Month (1-31)
          - Month (1-12)
          - Quarter (1-4)
          - Year
          - Is_Holiday (1 if public holiday, else 0)
          - Season (Winter, Spring, Summer, Fall)
          - Week_of_Year (1-52)
          - Days_Since_Start (days since the start of the dataset)
        """

        if "Date" not in df.columns:
            raise KeyError("The dataframe must contain a 'Date' column.")

        # Ensure 'Date' is in datetime format
        if df["Date"].dtype != "datetime64[ns]":
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

        # Extract date-based features
        df["Day_of_Week"] = df["Date"].dt.dayofweek  # 0=Monday, 6=Sunday
        df["Is_Weekend"] = (df["Day_of_Week"] >= 5).astype(int)  # 1 if Saturday/Sunday
        df["Day_of_Month"] = df["Date"].dt.day  # 1-31
        df["Day_of_Year"] = df["Date"].dt.dayofyear  # 1-365 (or 366 in leap years)
        df["Month"] = df["Date"].dt.month  # 1-12
        df["Quarter"] = df["Date"].dt.quarter  # 1-4
        df["Year"] = df["Date"].dt.year  # Extract Year
        df["Week_of_Year"] = df["Date"].dt.isocalendar().week  # Week number (1-52)
        df["Days_Since_Start"] = (df["Date"] - df["Date"].min()).dt.days  # Days since dataset start

        # Holiday feature
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=df["Date"].min(), end=df["Date"].max())
        df["Is_Holiday"] = df["Date"].isin(holidays).astype(int)

        # Define seasons
        def get_season(month):
            if month in [12, 1, 2]:
                return "Winter"
            elif month in [3, 4, 5]:
                return "Spring"
            elif month in [6, 7, 8]:
                return "Summer"
            else:
                return "Fall"

        df["Season"] = df["Month"].apply(get_season)

        return df







