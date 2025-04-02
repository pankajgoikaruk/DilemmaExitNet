import os
import logging
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler

# Adaptive max_points and max_levels

def crime_density(points, self):
    """
    Calculate max_points based on crime count variance.
    - High variance -> lower max_points (finer split in high-activity areas).
    - Low variance -> higher max_points (coarser split in uniform areas).
    """
    if not points:
        return 1000  # Default for empty nodes
    crime_counts = [p.Crime_count for p in points]
    variance = pd.Series(crime_counts).var()
    if pd.isna(variance):  # Handle NaN variance (0 or 1 point)
        logging.info(f"Node with {len(points)} points has NaN variance, using default max_points=1000")
        return 1000  # Default value when variance can’t be computed
    logging.info(f"Crime_count variance: {variance}")
    n_points_total = self.n_total # Use total dataset size.
    max_cap = max(1000, min(50000, int(n_points_total / 10)))  # Scale cap with dataset size
    # Base max_points between 500 and 2000, inversely proportional to variance
    # base_max = max(500, min(max_cap, int(5000 / (1 + variance / 2) + len(points))))  # Increasing 5000 value will increase the value of max_points. Increase base, reduce variance impact, increase point contribution
    base_max = max(500, min(max_cap, int(2000 / (1 + variance / 2) + len(
        points) / 2)))  # Reduced from 5000 to 2000, added len(points)/ to make max_points more adaptive to the actual number of points in the node, encouraging subdivision in denser areas.
    logging.info(f"Computed max_points: {base_max}, max_cap: {max_cap}")
    return base_max


def adaptive_max_levels(points, self):
    """
    Calculate max_levels based on crime count variance.
    - High variance -> higher max_levels (deeper tree for complex areas).
    - Low variance -> lower max_levels (shallower tree for uniform areas).
    """
    if not points:
        return 5  # Default for empty nodes
    crime_counts = [p.Crime_count for p in points]
    variance = pd.Series(crime_counts).var()
    if pd.isna(variance):  # Handle NaN variance
        return 5  # Default when variance can’t be computed
    n_points_total = self.n_total
    max_depth = max(5, min(25, int(math.log2(n_points_total) + 1)))  # Scale depth with log of size
    # Base max_levels, proportional to variance
    base_max = max(5, min(max_depth, int(5 + variance / 50 + math.log2(n_points_total) / 2)))
    logging.info(f"Computed max_levels: {base_max}, variance: {variance}, n_points_total: {n_points_total}")
    return base_max



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
        points = [Point(
            x=row['Longitude'], y=row['Latitude'], index=row['index'], Date=row['Date'], Time=row['Time'],
            Hour=row['Hour'], Minute=row['Minute'], Second=row['Second'], Scl_Longitude=row['Scl_Longitude'],
            Scl_Latitude=row['Scl_Latitude'], Day_of_Week=row['Day_of_Week'], Is_Weekend=row['Is_Weekend'],
            Day_of_Month=row['Day_of_Month'], Day_of_Year=row['Day_of_Year'], Month=row['Month'],
            Quarter=row['Quarter'], Year=row['Year'], Week_of_Year=row['Week_of_Year'],
            Days_Since_Start=row['Days_Since_Start'], Is_Holiday=row['Is_Holiday'], Season_Fall=row['Season_Fall'],
            Season_Spring=row['Season_Spring'], Season_Summer=row['Season_Summer'], Season_Winter=row['Season_Winter'],
            Crime_count=row['Crime_count'], Prediction=row['Prediction']
        ) for _, row in df.iterrows()]
        n_total = len(df)
        # initial_max_levels = adaptive_max_levels(points)
        # print(f"Initializing quadtree with adaptive max_points and max_levels.")
        # print(f"Computed initial max_levels based on data variance: {initial_max_levels}")
        boundary_rectangle = Rectangle(min(df['Longitude']), min(df['Latitude']), max(df['Longitude']),
                                       max(df['Latitude']))
        # initial_max_levels = adaptive_max_levels(points)  # Compute once
        quadtree = Quadtree(boundary_rectangle,
                            density_func=lambda p, self: crime_density(p, self),
                            max_levels_func=lambda p, self: adaptive_max_levels(p, self), # self is the Quadtree instance.
                            n_total=n_total) # Pass fixed initial_max_levels value
        inserted_count = 0
        for point in points:
            if quadtree.insert(point):
                inserted_count += 1
        logging.info(f"Total points inserted: {inserted_count} out of {n_total}")
        if hasattr(quadtree, 'max_depth'):
            logging.info(f"Maximum depth reached: {quadtree.max_depth}")
        return quadtree

    # @staticmethod
    # def init_quadtree(df):
    #     # Convert DataFrame to list of Points for initial max_levels calculation
    #     points = [Point(**row) for _, row in df.iterrows()]
    #     initial_max_levels = adaptive_max_levels(points)
    #
    #     print(f"Initializing quadtree with adaptive max_points and max_levels.")
    #     print(f"Computed initial max_levels based on data variance: {initial_max_levels}")
    #
    #     # Creates a boundary rectangle based on the min/max of Longitude and Latitude.
    #     boundary_rectangle = Rectangle(
    #         min(df['Longitude']), min(df['Latitude']),
    #         max(df['Longitude']), max(df['Latitude'])
    #     )
    #
    #     # Initialize quadtree with adaptive functions
    #     quadtree = Quadtree(boundary_rectangle, density_func=crime_density, max_levels_func=adaptive_max_levels)
    #
    #     # Iterates over the dataset and extracts relevant data points such as longitude, latitude, index, and other
    #     # features. Extract data points from Longitude and Latitude columns and insert them into the quadtree
    #     for label, row in df.iterrows():
    #         x = row['Longitude']
    #         y = row['Latitude']
    #         index = row['index']
    #         Date = row['Date']
    #         Time = row['Time']
    #         Hour = row['Hour']
    #         Minute = row['Minute']
    #         Second = row['Second']
    #         Scl_Longitude = row['Scl_Longitude']
    #         Scl_Latitude = row['Scl_Latitude']
    #         Day_of_Week = row['Day_of_Week']
    #         Is_Weekend = row['Is_Weekend']
    #         Day_of_Month = row['Day_of_Month']
    #         Day_of_Year = row['Day_of_Year']
    #         Month = row['Month']
    #         Quarter = row['Quarter']
    #         Year = row['Year']
    #         Week_of_Year = row['Week_of_Year']
    #         Days_Since_Start = row['Days_Since_Start']
    #         Is_Holiday = row['Is_Holiday']
    #         Season_Fall = row['Season_Fall']
    #         Season_Spring = row['Season_Spring']
    #         Season_Summer = row['Season_Summer']
    #         Season_Winter = row['Season_Winter']
    #         Crime_count = row['Crime_count']
    #         Prediction = row['Prediction']
    #
    #         # Creates a Point object for each data point with the extracted features and inserts it into the quadtree.
    #         point = Point(x, y, index, Date, Time, Hour, Minute, Second, Scl_Longitude, Scl_Latitude,
    #                       Day_of_Week, Is_Weekend, Day_of_Month, Day_of_Year, Month, Quarter, Year, Week_of_Year,
    #                       Days_Since_Start, Is_Holiday, Season_Fall, Season_Spring, Season_Summer, Season_Winter,
    #                       Crime_count, Prediction)
    #
    #         quadtree.insert(point)
    #
    #         # Returns the initialized quadtree.
    #     return quadtree


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
        return (self.x1 <= x <= self.x2) and (self.y1 <= y <= self.y2) # Exclude upper bounds to avoid overlap

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
    def __init__(self, boundary, max_points=None, max_levels=None, density_func=None, max_levels_func=None,
                 node_id=0, root_node=None, node_level=0, parent=None, df=None, ex_time=None, n_total=None):

        self.model = None  # To hold the current model while traversal through quadtree.
        self.boundary = boundary  # Stores the boundary rectangle of the quadtree.
        self.density_func = density_func if density_func is not None else crime_density
        self.max_levels_func = max_levels_func if max_levels_func is not None else adaptive_max_levels
        self.temp_points = []  # Stores the points that belong to the leaf nodes until they get subdivide.
        self.children = []  # Stores the child nodes of the quadtree.
        self.node_level = node_level  # Stores the level of the current node within the quadtree.
        self.node_id = node_id  # Assign a unique identifier to the root node
        self.points = []  # Stores the points that belong to the current node.
        self.parent = parent  # To hold the pointer of the parent node.
        self.df = df  # To store the current dataset while traversal though each node of quadtree.
        self.ex_time = ex_time  # To store execution time of each node.
        self.evaluation_results = []  # Initialize an empty list to store evaluation results
        self.n_total = n_total  # Store dataset size

        # Set adaptive values after points is defined
        self.max_points = max_points if max_points is not None else (
            density_func(self.points, self) if density_func else 1000)
        self.max_levels = max_levels if max_levels is not None else (
            max_levels_func(self.points, self) if max_levels_func else 5)

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

        # # Check if max_levels is exceeded before inserting
        # if self.node_level >= self.max_levels:
        #     logging.info(
        #         f"Node {self.node_id} at depth {self.node_level} reached max_levels {self.max_levels}, rejecting point")
        #     return False  # Stop insertion at this level

        # Check Boundary: Check if the point is within the boundary of the current node
        if not self.boundary.contains_point(point.x, point.y):
            logging.warning(f"Point ({point.x}, {point.y}) outside boundary of Node {self.node_id}")
            return False
        self.points.append(point)  # Appending entered data points to the parent nodes. 07/03/2025

        # Check Leaf Node: Check if the current node is a leaf node and there is space to insert the point
        if self.is_leaf():
            self.temp_points.append(point)  # Add point to temp_points
            # logging.info(f"Node ID: {self.node_id} set max_points to {self.max_points} with {len(self.points)} points, temp_points={len(self.temp_points)}")
            # logging.info(f"Node ID: {self.node_id} at current depth {self.node_level}, max_levels={self.max_levels}")

            if len(self.temp_points) >= self.max_points and self.node_level < self.max_levels:
                self.subdivide()
            return True
            # if len(self.temp_points) < self.max_points:
            #     self.temp_points.append(point)
            #     return True
            # if self.node_level < self.max_levels:  # Explicit max_levels check
            #     self.subdivide() # Subdivide if capacity exceeded

        # # Subdivide Node: If the current node is a leaf node, or it's full, subdivide it
        # if not self.children and self.node_level < self.max_levels:
        #     self.subdivide()

        # Insert into Child Nodes: Attempt to insert the point into the child nodes
        inserted = False
        for child in self.children:
            if child.boundary.contains_point(point.x, point.y):
                inserted = child.insert(point, child.node_id)  # Pass current node ID to child
                break
        # return False  # If no child can accept the point
        if not inserted:
            logging.warning(
                f"Point ({point.x}, {point.y}) not inserted into any child of Node {self.node_id}, storing in current node")
            self.temp_points.append(point)  # Store in current node if no child accepts it
        return True

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

        # Update max_points frequency based on dataset size
        update_frequency = 3 if self.n_total > 1000000 else 1  # Every 3 levels for large datasets, every level for small
        update_max_points = self.node_level % update_frequency == 0
        child_max_points = self.density_func(self.points, self) if update_max_points else self.max_points
        child_max_levels = self.max_levels_func(self.points, self) if update_max_points else self.max_levels  # Update max_levels
        if update_max_points:
            logging.info(f"Node {self.node_id} at depth {self.node_level} updating max_points to {child_max_points}, max_levels to {child_max_levels}")

        # Create child nodes for each quadrant.
        for boundary in quadrant_boundaries:
            self.root_node.global_count += 1
            child = Quadtree(
                boundary,
                max_points=child_max_points,  # self.max_points Initial value, will be updated by density_func
                max_levels=child_max_levels,  # self.max_levels
                density_func=self.density_func,  # Pass density function to children
                max_levels_func=self.max_levels_func,  # Pass adaptive quadtree level function to children
                node_id=self.root_node.global_count,
                root_node=self.root_node,
                parent=self,
                node_level=self.node_level + 1,
                n_total = self.n_total  # Pass to children
            )
            self.children.append(child)
            # Update max depth seen
            if hasattr(self.root_node, 'max_depth'):
                self.root_node.max_depth = max(self.root_node.max_depth, child.node_level)
            else:
                self.root_node.max_depth = child.node_level
            logging.info(
                f"Node {child.node_id} created at current node level {child.node_level}, computed max_levels={child_max_levels}, assigned max_levels={child.max_levels}")
            # logging.info(f"Node {child.node_id} created at current node level {child.node_level}, computed max_levels={child_max_levels}, assigned max_levels={child.max_levels}")

        # print(f"Splitting at depth {self.node_level}, max_levels: {self.max_levels}")

        # Distribute points to children (Insert all points stored in temp_points into the appropriate child).
        for point in self.temp_points:
            inserted = False
            for child in self.children:
                if child.boundary.contains_point(point.x, point.y):
                    child.insert(point)
                    inserted = True
                    break
            if not inserted:
                logging.warning(
                    f"Point ({point.x}, {point.y}) not inserted into any child node during subdivision of Node {self.node_id}")

        # Clear temp_points as they have been distributed.
        self.temp_points = []
        self.max_points = self.density_func(self.points, self) if update_max_points else self.max_points
        self.max_levels = self.max_levels_func(self.points, self) if update_max_points else self.max_levels
        # self.max_points = self.density_func(self.points)  # More frequently check the node density.
        # self.max_levels = self.max_levels_func(self.points)

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

    # Convert a datetime format to Unix timestamp.u,bkh
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
