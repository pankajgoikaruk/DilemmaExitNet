import os
import logging
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler


def setup_directories(dir_list):
    """
    Create directories if they do not exist.
    """
    for directory in dir_list:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created directory: {directory}")


# Define directories.
dcr_dir_csv = 'node_pred_dir_csv'
output_dir_csv = 'output_csv'
model_saved = 'model_saved'
setup_directories([dcr_dir_csv, output_dir_csv, model_saved])


class InitialQuadtree:
    def __init__(self) -> None:
        self.evaluation_results = []  # Initialize an empty list to store evaluation results

    # Modeling to generated prediction values.
    @staticmethod
    def set_pred_zero(df):
        df = df.copy()  # Prevent modification on a slice
        df.loc[:, 'Date'] = Quadtree.datetime_to_unix_timestamps(df)
        df.loc[:, 'Crime_count'] = Quadtree.min_max_scale_values(df, col_name='Crime_count').round()
        df.loc[:, 'Prediction'] = 0
        return df


    @staticmethod
    def init_quadtree(df):
        # n = len(df)
        # # Heuristic for max_points: use square root of n, with a minimum value.
        # default_max_points = max(4, int(math.sqrt(n)))
        #
        # print(f"Recommended maximum points per node: {default_max_points}")
        #
        # # Prompt the user for maximum number of points per node.
        # while True:
        #     try:
        #         user_input = input("Enter the maximum number of points per node [Press Enter to accept default]: ")
        #         if user_input == "":
        #             max_points = default_max_points
        #         else:
        #             max_points = int(user_input)
        #             if max_points < default_max_points:
        #                 print(f"Please enter a value greater than or equal to {default_max_points}.")
        #                 continue
        #         break
        #     except ValueError:
        #         print("Please enter a positive integer value.")
        #
        # # Now compute default_max_levels based on worst-case scenario:
        # default_max_levels = max(3, math.ceil(n / max_points))
        # print(f"Recommended maximum quadtree levels (worst-case): {default_max_levels}")
        #
        # # Prompt the user for maximum number of quadtree levels.
        # while True:
        #     try:
        #         user_input = input(
        #             "Enter the maximum number of levels in the quadtree [Press Enter to accept default]: ")
        #         if user_input == "":
        #             max_levels = default_max_levels
        #         else:
        #             max_levels = int(user_input)
        #             if max_levels <= 1:
        #                 raise ValueError
        #         break
        #     except ValueError:
        #         print("Please enter a positive integer greater than 1 for maximum levels.")

        n = len(df)
        # Alternative heuristic: use a fraction of n for max_points
        default_max_points = max(1000,
                                 int(n / 50))  # The default maximum number of points per node is the larger of 100 and 𝑛/500
        # (rounded down). The default maximum levels is the larger of 3 and the
        #  ceiling of the ratio 𝑛/default_max_points.

        # Compute default_max_levels based on worst-case scenario
        default_max_levels = max(5, math.ceil(n / default_max_points))

        print(f"Recommended maximum points per node: {default_max_points}")
        print(f"Recommended maximum quadtree levels: {default_max_levels}")

        # Prompt the user for maximum number of points per node, with a default recommendation.
        while True:
            try:
                user_input = input("Enter the maximum number of points per node [Press Enter to accept default]: ")
                if user_input == "":
                    max_points = default_max_points
                else:
                    max_points = int(user_input)
                    if max_points < default_max_points:
                        print(f"Please enter a value greater than or equal to {default_max_points}.")
                        continue
                break
            except ValueError:
                print("Please enter a positive integer value.")

        # Prompt the user for maximum number of quadtree levels, with a default recommendation.
        while True:
            try:
                user_input = input(
                    "Enter the maximum number of levels in the quadtree [Press Enter to accept default]: ")
                if user_input == "":
                    max_levels = default_max_levels
                else:
                    max_levels = int(user_input)
                    if max_levels <= 1:
                        raise ValueError
                break
            except ValueError:
                print("Please enter a positive integer greater than 1 for maximum levels.")

        # Creates a boundary rectangle based on the min/max of Longitude and Latitude.
        boundary_rectangle = Rectangle(
            min(df['Longitude']), min(df['Latitude']),
            max(df['Longitude']), max(df['Latitude'])
        )

        # Initializes the Quadtree object.
        quadtree = Quadtree(boundary_rectangle, max_points, max_levels)
        # return quadtree

        # Iterates over the dataset and extracts relevant data points such as longitude, latitude, index, and other
        # features. Extract data points from Longitude and Latitude columns and insert them into the quadtree
        for label, row in df.iterrows():
            x = row['Longitude']
            y = row['Latitude']
            index = row['index']
            Date = row['Date']
            Time = row['Time']
            Hour = row['Hour']
            Minute = row['Minute']
            Second = row['Second']
            Scl_Longitude = row['Scl_Longitude']
            Scl_Latitude = row['Scl_Latitude']
            Day_of_Week = row['Day_of_Week']
            Is_Weekend = row['Is_Weekend']
            Day_of_Month = row['Day_of_Month']
            Day_of_Year = row['Day_of_Year']
            Month = row['Month']
            Quarter = row['Quarter']
            Year = row['Year']
            Week_of_Year = row['Week_of_Year']
            Days_Since_Start = row['Days_Since_Start']
            Is_Holiday = row['Is_Holiday']
            Season_Fall = row['Season_Fall']
            Season_Spring = row['Season_Spring']
            Season_Summer = row['Season_Summer']
            Season_Winter = row['Season_Winter']
            Crime_count = row['Crime_count']
            Prediction = row['Prediction']

            # Creates a Point object for each data point with the extracted features and inserts it into the quadtree.
            point = Point(x, y, index, Date, Time, Hour, Minute, Second, Scl_Longitude, Scl_Latitude,
                          Day_of_Week, Is_Weekend, Day_of_Month, Day_of_Year, Month, Quarter, Year, Week_of_Year,
                          Days_Since_Start, Is_Holiday, Season_Fall, Season_Spring, Season_Summer, Season_Winter,
                          Crime_count, Prediction)

            quadtree.insert(point)

            # Returns the initialized quadtree.
        return quadtree


# Represents a point with various attributes such as coordinates (x and y) and additional information related to
# crime data (Date, Scl_Longitude, etc.).
class Point:
    def __init__(self, x, y, index, Date, Time, Hour, Minute, Second, Scl_Longitude, Scl_Latitude,
                 Day_of_Week, Is_Weekend, Day_of_Month, Day_of_Year, Month, Quarter, Year, Week_of_Year,
                 Days_Since_Start, Is_Holiday, Season_Fall, Season_Spring, Season_Summer, Season_Winter,
                 Crime_count, Prediction):
        self.x = x  # Longitude
        self.y = y  # Latitude
        self.index = index
        self.Date = Date
        self.Time = Time
        self.Hour = Hour
        self.Minute = Minute
        self.Second = Second
        self.Scl_Longitude = Scl_Longitude
        self.Scl_Latitude = Scl_Latitude
        self.Day_of_Week = Day_of_Week
        self.Is_Weekend = Is_Weekend
        self.Day_of_Month = Day_of_Month
        self.Day_of_Year = Day_of_Year
        self.Month = Month
        self.Quarter = Quarter
        self.Year = Year
        self.Week_of_Year = Week_of_Year
        self.Days_Since_Start = Days_Since_Start
        self.Is_Holiday = Is_Holiday
        self.Season_Fall = Season_Fall
        self.Season_Spring = Season_Spring
        self.Season_Summer = Season_Summer
        self.Season_Winter = Season_Winter
        self.Crime_count = Crime_count
        self.Prediction = Prediction


""" 
Represents a rectangle with bottom-left and top-right corner coordinates. It provides methods to check if a point
lies within the rectangle (contains_point) and if it intersects with another rectangle (intersects).
"""


class Rectangle:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    # Check if a point lies within the rectangle. Returns True if the point lies within the rectangle, False otherwise.
    def contains_point(self, x, y):
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    # Check if the rectangle intersects with another rectangle.
    def intersects(self, other):
        return not (self.x2 < other.x1 or self.x1 > other.x2 or self.y2 < other.y1 or self.y1 > other.y2)


"""
Represents the quadtree data structure. It is initialized with a boundary rectangle, maximum number of points per 
node (max_points), and maximum depth (max_levels). The quadtree is recursively divided into four quadrants until each 
quadrant contains no more than max_points or reaches the maximum depth. It provides methods to insert points into the 
quadtree (insert), subdivide a node into quadrants (subdivide), and check if a node is a leaf node (is_leaf).
"""


class Quadtree:
    def __init__(self, boundary, max_points=None, max_levels=None, node_id=0,
                 root_node=None,
                 node_level=0,
                 parent=None,
                 df=None,
                 ex_time=None):

        self.model = None  # To hold the current model while traversal through quadtree.
        self.boundary = boundary  # Stores the boundary rectangle of the quadtree.
        self.max_points = max_points if max_points is not None else 4
        self.max_levels = max_levels if max_levels is not None else 10
        self.temp_points = []  # Stores the points that belong to the leaf nodes until they get subdivide.
        self.children = []  # Stores the child nodes of the quadtree.
        self.node_level = node_level  # Stores the level of the current node within the quadtree.
        self.node_id = node_id  # Assign a unique identifier to the root node
        self.points = []  # Stores the points that belong to the current node.
        self.parent = parent  # To hold the pointer of the parent node.
        self.df = df  # To store the current dataset while traversal though each node of quadtree.
        self.ex_time = ex_time  # To store execution time of each node.
        self.evaluation_results = []  # Initialize an empty list to store evaluation results

        # Node IDs assignment.
        if root_node is None:
            self.root_node = self  # Assigning root in itself.
            self.global_count = 0  # Set global counter to keep sequence and track on node ids.
        else:
            self.root_node = root_node  # Setting current node id.

        # Ensure that boundary is a Rectangle object
        if not isinstance(self.boundary, Rectangle):
            raise ValueError("Boundary must be a Rectangle object")

    def insert(self, point, node_id=0):  # Added node_id argument for recursive calls
        # Check if max_levels is exceeded before inserting
        if self.node_level >= self.max_levels:
            return False  # Stop insertion at this level

        # Check Boundary: Check if the point is within the boundary of the current node
        if not self.boundary.contains_point(point.x, point.y):
            return False
        self.points.append(point)  # Appending entered data points to the parent nodes. 07/03/2025

        # Check Leaf Node: Check if the current node is a leaf node and there is space to insert the point
        if self.is_leaf() and len(self.temp_points) < self.max_points:
            self.temp_points.append(point)
            return True

        # Subdivide Node: If the current node is a leaf node, or it's full, subdivide it
        if not self.children and self.node_level < self.max_levels:
            self.subdivide()

        # Insert into Child Nodes: Attempt to insert the point into the child nodes
        for child in self.children:
            if child.boundary.contains_point(point.x, point.y):
                child.insert(point, child.node_id)  # Pass current node ID to child

        return False  # If no child can accept the point

    def subdivide(self):

        # Calculate the dimensions of each child node
        x_mid = (self.boundary.x1 + self.boundary.x2) / 2
        y_mid = (self.boundary.y1 + self.boundary.y2) / 2

        # # Create child nodes representing each quadrant
        # node_id_counter = self.root_node.global_count + 1  # Increasing global count for each node.
        # self.root_node.global_count = node_id_counter  # Assigned in local variable.
        # nw_boundary = Rectangle(self.boundary.x1, y_mid, x_mid, self.boundary.y2)
        # nw_quadtree = Quadtree(nw_boundary, self.max_points, self.max_levels, root_node=self.root_node,
        #                        parent=self)  # tree_level=self.tree_level,
        # nw_quadtree.node_id = node_id_counter  # Assigned id to the current node.
        # # nw_quadtree.node_level = tree_level_count
        # nw_quadtree.node_level = self.node_level + 1
        # self.children.append(nw_quadtree)
        #
        # node_id_counter = self.root_node.global_count + 1
        # self.root_node.global_count = node_id_counter
        # ne_boundary = Rectangle(x_mid, y_mid, self.boundary.x2, self.boundary.y2)
        # ne_quadtree = Quadtree(ne_boundary, self.max_points, self.max_levels, root_node=self.root_node,
        #                        parent=self)  # tree_level=self.tree_level,
        # ne_quadtree.node_id = node_id_counter
        # ne_quadtree.node_level = self.node_level + 1
        # self.children.append(ne_quadtree)
        #
        # node_id_counter = self.root_node.global_count + 1
        # self.root_node.global_count = node_id_counter
        # sw_boundary = Rectangle(self.boundary.x1, self.boundary.y1, x_mid, y_mid)
        # sw_quadtree = Quadtree(sw_boundary, self.max_points, self.max_levels, root_node=self.root_node,
        #                        parent=self)  # tree_level=self.tree_level,
        # sw_quadtree.node_id = node_id_counter
        # sw_quadtree.node_level = self.node_level + 1
        # self.children.append(sw_quadtree)
        #
        # node_id_counter = self.root_node.global_count + 1
        # self.root_node.global_count = node_id_counter
        # se_boundary = Rectangle(x_mid, self.boundary.y1, self.boundary.x2, y_mid)
        # se_quadtree = Quadtree(se_boundary, self.max_points, self.max_levels, root_node=self.root_node,
        #                        parent=self)  # tree_level=self.tree_level,
        # se_quadtree.node_id = node_id_counter
        # se_quadtree.node_level = self.node_level + 1
        # self.children.append(se_quadtree)

        # Define the boundaries for each quadrant.
        quadrant_boundaries = [
            Rectangle(self.boundary.x1, y_mid, x_mid, self.boundary.y2),  # NW
            Rectangle(x_mid, y_mid, self.boundary.x2, self.boundary.y2),  # NE
            Rectangle(self.boundary.x1, self.boundary.y1, x_mid, y_mid),  # SW
            Rectangle(x_mid, self.boundary.y1, self.boundary.x2, y_mid)  # SE
        ]

        # Create child nodes for each quadrant.
        for boundary in quadrant_boundaries:
            self.root_node.global_count += 1
            child = Quadtree(
                boundary,
                self.max_points,
                self.max_levels,
                node_id=self.root_node.global_count,
                root_node=self.root_node,
                parent=self,
                node_level=self.node_level + 1
            )
            self.children.append(child)

        # print(f"Splitting at depth {self.node_level}, max_levels: {self.max_levels}")

        # Distribute points to children (Insert all points stored in temp_points into the appropriate child).
        for point in self.temp_points:
            for child in self.children:
                if child.boundary.contains_point(point.x, point.y):
                    child.insert(point)
                    break

        # Clear temp_points as they have been distributed.
        self.temp_points = []

    # Check if the current node is a leaf node (i.e., it has no children).
    def is_leaf(self):
        return len(self.children) == 0

    # # Recursive method to traversal through quadtree using Depth-First Search Algorithm.
    # def traverse_quadtree_modelling(self):
    #     if self.node_id == 0:
    #         print(f"Visiting node ID: {self.node_id}")
    #     self.modelling()
    #     for child in self.children:
    #         print(f"Visiting node ID: {child.node_id}")
    #         print()
    #         child.traverse_quadtree_modelling()
    #         print(f"Finished visiting node ID: {child.node_id}")
    #         print()

    # Convert a datetime format to Unix timestamp.
    @staticmethod
    def datetime_to_unix_timestamps(df):
        # df['Date'] = df['Date'].astype('int64') // 10 ** 9
        df = df.copy()  # Avoid modifying a view
        df.loc[:, 'Date'] = pd.to_datetime(df['Date'])  # Ensure Date is in datetime format
        df.loc[:, 'Date'] = df['Date'].astype('int64') // 10 ** 9  # Convert to Unix timestamp explicitly
        df.loc[:, 'Date'] = df['Date'].astype('int64')  # Ensure final dtype is int64
        return df['Date']

    # Scale down the target and predicted values
    @staticmethod
    def min_max_scale_values(df, col_name):
        df = df.copy()  # Prevent modification on a slice
        # Reshape the column values to a 2D array
        col_counts = df[col_name].values.reshape(-1,
                                                 1)  # -1 means NumPy figure out the number of the rows automatically.
        # 1 means keep a single column.
        # Initialize the scaler
        min_max_scaler = MinMaxScaler(feature_range=(100, 105))

        # Fit and transform the column values
        # df[col_name] = min_max_scaler.fit_transform(col_counts)
        df.loc[:, col_name] = min_max_scaler.fit_transform(col_counts).astype(int)

        # Return the scaled column (avoiding direct modification of df)
        return df[col_name]  # Convert back to 1D
