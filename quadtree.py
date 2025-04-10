# import os
# import csv
# import time
# import logging
# import pandas as pd
# import math
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import json
#
# # Define directories.
# node_dcr = 'node_dcr'
# dcr_dir_csv = 'node_pred_dir_csv'
# output_dir_csv = 'output_csv'
# model_saved = 'model_saved'
#
# def setup_directories(dir_list):
#     """
#     Create directories if they do not exist.
#     """
#     for directory in dir_list:
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#             logging.info(f"Created directory: {directory}")
#
# # Set up the directories
# setup_directories([node_dcr, dcr_dir_csv, output_dir_csv, model_saved])
#
#
# # Reference dataset size and tuned values
# REFERENCE_SIZE = 1140125
# TUNED_VALUES = {
#     "alpha": 5000,  # Increased from 1000
#     "kappa": 100000,  # Increased from 50000
#     "lambda_val": 10,
#     "min_base": 2000,  # Increased from 500
#     "beta": 10000,  # Increased from 2000
#     "gamma": 2,
#     "delta": 2
# }
#
#
# def calculate_constants(dataset_size):
#     if not isinstance(dataset_size, (int, float)):
#         raise TypeError(f"Expected dataset_size to be a number, got {type(dataset_size)}")
#
#     # Calculate scaling factor
#     scaling_factor = dataset_size / REFERENCE_SIZE
#
#     # Initialize new constants
#     new_constants = {}
#
#     # Linear scaling for alpha, kappa, min_base, beta
#     for param in ["alpha", "kappa", "min_base", "beta"]:
#         new_constants[param] = max(1, round(TUNED_VALUES[param] * scaling_factor))
#
#     # Logarithmic scaling for lambda_val, gamma, delta with bounds
#     log_scale = 1 + math.log(max(1, scaling_factor)) if scaling_factor > 0 else 1
#
#     # lambda_val: Keep between 5 and 20
#     new_constants["lambda_val"] = min(20, max(5, round(TUNED_VALUES["lambda_val"] * log_scale)))
#
#     # gamma: Keep between 1 and 5
#     new_constants["gamma"] = min(5, max(1, round(TUNED_VALUES["gamma"] * log_scale)))
#
#     # delta: Keep between 1 and 5
#     new_constants["delta"] = min(5, max(1, round(TUNED_VALUES["delta"] * log_scale)))
#
#     return new_constants
#
#
# def crime_density(points, self):
#     """
#     #     Calculate max_points based on crime count variance.
#     #     - High variance -> lower max_points (finer split in high-activity areas).
#     #     - Low variance -> higher max_points (coarser split in uniform areas).
#     #     """
#
#     if not points:
#         return self.alpha  # e.g., 1000
#
#     crime_counts = [p.Crime_count for p in points]
#     variance = pd.Series(crime_counts).var()
#     if pd.isna(variance):
#         logging.info(f"Node with {len(points)} points has NaN variance, using default max_points={self.alpha}")
#         return self.alpha
#     logging.info(f"Crime_count variance: {variance}")
#     n_points_total = self.n_total
#     max_cap = max(self.alpha, min(self.kappa, int(n_points_total / self.lambda_val)))  # e.g., alpha=1000, kappa=50000, lambda_val=10
#     base_max = max(self.min_base, min(max_cap, int(self.beta / (1 + variance / self.gamma) + len(points) / self.delta)))  # e.g., min_base=500, beta=2000, gamma=2, delta=2
#     logging.info(f"Computed max_points: {base_max}, max_cap: {max_cap}")
#     return base_max
#
#
# def adaptive_max_levels(points, self):
#     """
#     Calculate max_levels adaptively based on dataset size and crime count variance.
#     - High variance -> higher max_levels (deeper tree for complex areas).
#     - Low variance -> lower max_levels (shallower tree for uniform areas).
#     - Scales with log of dataset size and variance for deeper trees in complex datasets.
#     - Uses an adaptive cap (mu) to ensure scalability for very large datasets.
#     """
#     if not points:
#         return 5  # Default for empty nodes
#
#     # Total number of points
#     n_points_total = self.n_total
#
#     # Compute variance of Crime_count
#     crime_counts = [p.Crime_count for p in points]
#     variance = pd.Series(crime_counts).var() if crime_counts else 0.0
#     if pd.isna(variance):  # Handle NaN variance
#         return 5  # Default when variance can’t be computed
#
#     # Adaptive mu based on dataset size
#     eta = 1.5  # Scaling factor
#     mu = int(eta * math.log2(n_points_total)) if n_points_total > 0 else 5
#
#     # Compute max_levels: combine log of dataset size and variance, capped by mu
#     computed_levels = int(math.log2(n_points_total) + 1 + variance) if n_points_total > 0 else 5
#     max_levels = min(mu, computed_levels)
#
#     # Hard cap at 10
#     max_levels = min(15, max_levels)
#
#     # Ensure a minimum depth for small datasets
#     max_levels = max(5, max_levels)
#
#     logging.info(f"Computed max_levels: {max_levels}, variance: {variance}, n_points_total: {n_points_total}")
#     return max_levels
#
#
# class InitialQuadtree:
#     def __init__(self) -> None:
#         self.evaluation_results = []  # Initialize an empty list to store evaluation results
#
#     # Modeling to generated prediction values.
#     @staticmethod
#     def set_pred_zero(df):
#         df = df.copy()  # Prevent modification on a slice
#         df.loc[:, 'Date'] = Quadtree.datetime_to_unix_timestamps(df)
#         df.loc[:, 'Crime_count'] = Quadtree.min_max_scale_values(df, col_name='Crime_count').round()
#         df.loc[:, 'Prediction'] = 0
#         return df
#
#     @staticmethod
#     def init_quadtree(df, constants):
#         points = [Point(
#             x=row['Longitude'], y=row['Latitude'], index=row['index'], Date=row['Date'], Time=row['Time'],
#             Hour=row['Hour'], Minute=row['Minute'], Second=row['Second'], Scl_Longitude=row['Scl_Longitude'],
#             Scl_Latitude=row['Scl_Latitude'], Day_of_Week=row['Day_of_Week'], Is_Weekend=row['Is_Weekend'],
#             Day_of_Month=row['Day_of_Month'], Day_of_Year=row['Day_of_Year'], Month=row['Month'],
#             Quarter=row['Quarter'], Year=row['Year'], Week_of_Year=row['Week_of_Year'],
#             Days_Since_Start=row['Days_Since_Start'], Is_Holiday=row['Is_Holiday'], Season_Fall=row['Season_Fall'],
#             Season_Spring=row['Season_Spring'], Season_Summer=row['Season_Summer'], Season_Winter=row['Season_Winter'],
#             Crime_count=row['Crime_count'], Prediction=row['Prediction']
#         ) for _, row in df.iterrows()]
#         n_total = len(df)
#
#         boundary_rectangle = Rectangle(min(df['Longitude']), min(df['Latitude']), max(df['Longitude']),
#                                        max(df['Latitude']))
#
#         quadtree = Quadtree(boundary_rectangle,
#                             # points=df[['Longitude', 'Latitude']].values,
#                             density_func=lambda p, self: crime_density(p, self),
#                             max_levels_func=lambda p, self: adaptive_max_levels(p, self), # self is the Quadtree instance.
#                             n_total=n_total,
#                             alpha = constants["alpha"],
#                             kappa = constants["kappa"],
#                             lambda_val = constants["lambda_val"],
#                             min_base = constants["min_base"],
#                             beta = constants["beta"],
#                             gamma = constants["gamma"],
#                             delta = constants["delta"]) # Pass fixed initial_max_levels value
#         inserted_count = 0
#         for point in points:
#             if quadtree.insert(point):
#                 inserted_count += 1
#         logging.info(f"Total points inserted: {inserted_count} out of {n_total}")
#         if hasattr(quadtree, 'max_depth'):
#             logging.info(f"Maximum depth reached: {quadtree.max_depth}")
#
#         # # Log the details of all nodes in the Quadtree
#         # logging.info("Logging details of all nodes in the Quadtree:")
#         # quadtree.traverse_quadtree()
#
#         # Perform merging of small leaf nodes
#         quadtree.merge_small_leaf_nodes(threshold=1000)
#
#         return quadtree
#
#
# # Represents a point with various attributes such as coordinates (x and y) and additional information related to
# # crime data (Date, Scl_Longitude, etc.).
# class Point:
#     def __init__(self, x, y, index, Date, Time, Hour, Minute, Second, Scl_Longitude, Scl_Latitude,
#                  Day_of_Week, Is_Weekend, Day_of_Month, Day_of_Year, Month, Quarter, Year, Week_of_Year,
#                  Days_Since_Start, Is_Holiday, Season_Fall, Season_Spring, Season_Summer, Season_Winter,
#                  Crime_count, Prediction):
#         self.x = x  # Longitude
#         self.y = y  # Latitude
#         self.index = index
#         self.Date = Date
#         self.Time = Time
#         self.Hour = Hour
#         self.Minute = Minute
#         self.Second = Second
#         self.Scl_Longitude = Scl_Longitude
#         self.Scl_Latitude = Scl_Latitude
#         self.Day_of_Week = Day_of_Week
#         self.Is_Weekend = Is_Weekend
#         self.Day_of_Month = Day_of_Month
#         self.Day_of_Year = Day_of_Year
#         self.Month = Month
#         self.Quarter = Quarter
#         self.Year = Year
#         self.Week_of_Year = Week_of_Year
#         self.Days_Since_Start = Days_Since_Start
#         self.Is_Holiday = Is_Holiday
#         self.Season_Fall = Season_Fall
#         self.Season_Spring = Season_Spring
#         self.Season_Summer = Season_Summer
#         self.Season_Winter = Season_Winter
#         self.Crime_count = Crime_count
#         self.Prediction = Prediction
#
#
# """
# Represents a rectangle with bottom-left and top-right corner coordinates. It provides methods to check if a point
# lies within the rectangle (contains_point) and if it intersects with another rectangle (intersects).
# """
#
#
# class Rectangle:
#     def __init__(self, x1, y1, x2, y2):
#         self.x1 = x1
#         self.y1 = y1
#         self.x2 = x2
#         self.y2 = y2
#
#     # Check if a point lies within the rectangle. Returns True if the point lies within the rectangle, False otherwise.
#     def contains_point(self, x, y):
#         return (self.x1 <= x <= self.x2) and (self.y1 <= y <= self.y2) # Exclude upper bounds to avoid overlap
#
#     # Check if the rectangle intersects with another rectangle.
#     def intersects(self, other):
#         return not (self.x2 < other.x1 or self.x1 > other.x2 or self.y2 < other.y1 or self.y1 > other.y2)
#
#
# """
# Represents the quadtree data structure. It is initialized with a boundary rectangle, maximum number of points per
# node (max_points), and maximum depth (max_levels). The quadtree is recursively divided into four quadrants until each
# quadrant contains no more than max_points or reaches the maximum depth. It provides methods to insert points into the
# quadtree (insert), subdivide a node into quadrants (subdivide), and check if a node is a leaf node (is_leaf).
# """
#
#
# class Quadtree:
#     def __init__(self, boundary, max_points=None, max_levels=None, density_func=None, max_levels_func=None,
#                  node_id=0, root_node=None, node_level=0, parent=None, df=None, ex_time=None, n_total=None,
#                  alpha=None, kappa=None, lambda_val=None, min_base=None, beta=None, gamma=None, delta=None): #points,
#
#         self.model = None  # To hold the current model while traversal through quadtree.
#         self.boundary = boundary  # Stores the boundary rectangle of the quadtree.
#         self.density_func = density_func if density_func is not None else crime_density
#         self.max_levels_func = max_levels_func if max_levels_func is not None else adaptive_max_levels
#         self.temp_points = []  # Stores the points that belong to the leaf nodes until they get subdivide.
#         self.children = []  # Stores the child nodes of the quadtree.
#         self.node_level = node_level  # Stores the level of the current node within the quadtree.
#         self.node_id = node_id  # Assign a unique identifier to the root node
#         self.points = []  # Stores the points that belong to the current node.
#         self.parent = parent  # To hold the pointer of the parent node.
#         self.df = df  # To store the current dataset while traversal though each node of quadtree.
#         self.ex_time = ex_time  # To store execution time of each node.
#         self.evaluation_results = []  # Initialize an empty list to store evaluation results
#         self.n_total = n_total  # Store dataset size
#
#         # Symbolic constants with default values
#         self.alpha = alpha  # Default for empty nodes or NaN variance
#         self.kappa = kappa  # Upper cap for max_points
#         self.lambda_val = lambda_val  # Scaling factor for max_cap
#         self.min_base = min_base  # Lower bound for base_max
#         self.beta = beta  # Base value for variance scaling
#         self.gamma = gamma  # Variance scaling factor
#         self.delta = delta  # Point count scaling factor
#
#         self.merged_pairs = {}  # To store merge mappings
#
#         # Validate constants
#         required_constants = ['alpha', 'kappa', 'lambda_val', 'min_base', 'beta', 'gamma', 'delta']
#         for const in required_constants:
#             if getattr(self, const) is None:
#                 raise ValueError(f"Constant '{const}' must be provided and cannot be None")
#
#         # Set adaptive values after points is defined
#         self.max_points = max_points if max_points is not None else (
#             density_func(self.points, self) if density_func else 1000)
#         self.max_levels = max_levels if max_levels is not None else (
#             max_levels_func(self.points, self) if max_levels_func else 5)
#
#         # Node IDs assignment.
#         if root_node is None:
#             self.root_node = self  # Assigning root in itself.
#             self.global_count = 0  # Set global counter to keep sequence and track on node ids.
#         else:
#             self.root_node = root_node  # Setting current node id.
#
#         # Ensure that boundary is a Rectangle object
#         if not isinstance(self.boundary, Rectangle):
#             raise ValueError("Boundary must be a Rectangle object")
#
#     def insert(self, point, node_id=0):  # Added node_id argument for recursive calls
#
#         # If node_id is provided, use it; otherwise, use self.node_id
#         if node_id is None:
#             node_id = self.node_id
#
#         # Check Boundary: Check if the point is within the boundary of the current node
#         if not self.boundary.contains_point(point.x, point.y):
#             logging.warning(f"Point ({point.x}, {point.y}) outside boundary of Node {self.node_id}")
#             return False
#
#         self.points.append(point)  # Appending entered data points to the parent nodes. 07/03/2025
#
#         # Check Leaf Node: Check if the current node is a leaf node and there is space to insert the point
#         if self.is_leaf():
#             self.temp_points.append(point)  # Add point to temp_points
#             # logging.info(f"Node ID: {self.node_id} set max_points to {self.max_points} with {len(self.points)} points, temp_points={len(self.temp_points)}")
#             # logging.info(f"Node ID: {self.node_id} at current depth {self.node_level}, max_levels={self.max_levels}")
#
#             if len(self.temp_points) >= self.max_points and self.node_level < self.max_levels:
#                 self.subdivide()
#             else:
#                 # Update max depth seen at the root level
#                 if hasattr(self.root_node, 'max_depth'):
#                     self.root_node.max_depth = max(self.root_node.max_depth, self.node_level)
#                 else:
#                     self.root_node.max_depth = self.node_level
#                 return True
#
#         # Insert into Child Nodes: Attempt to insert the point into the child nodes
#         inserted = False
#         for child in self.children:
#             if child.boundary.contains_point(point.x, point.y):
#                 inserted = child.insert(point, child.node_id)  # Pass current node ID to child
#                 if inserted:
#                     break
#
#         # If the point wasn't inserted into any child, store it in the current node's temp_points
#         if not inserted:
#             logging.warning(
#                 f"Point ({point.x}, {point.y}) not inserted into any child of Node {self.node_id}, storing in current node")
#             self.temp_points.append(point)  # Store in current node if no child accepts it
#         return True
#
#     # def subdivide(self):
#     #
#     #     # Calculate the dimensions of each child node
#     #     x_mid = (self.boundary.x1 + self.boundary.x2) / 2
#     #     y_mid = (self.boundary.y1 + self.boundary.y2) / 2
#     #
#     #     # Define the boundaries for each quadrant.
#     #     quadrant_boundaries = [
#     #         Rectangle(self.boundary.x1, y_mid, x_mid, self.boundary.y2),  # NW
#     #         Rectangle(x_mid, y_mid, self.boundary.x2, self.boundary.y2),  # NE
#     #         Rectangle(self.boundary.x1, self.boundary.y1, x_mid, y_mid),  # SW
#     #         Rectangle(x_mid, self.boundary.y1, self.boundary.x2, y_mid)  # SE
#     #     ]
#     #
#     #     # Update max_points frequency based on dataset size
#     #     update_frequency = 3 if self.n_total > 1000000 else 1  # Every 3 levels for large datasets, every level for small
#     #     update_max_points = self.node_level % update_frequency == 0
#     #
#     #     child_max_points = self.density_func(self.points, self) if update_max_points else self.max_points
#     #     child_max_levels = self.max_levels_func(self.points, self) if update_max_points else self.max_levels  # Update max_levels
#     #
#     #     if update_max_points:
#     #         logging.info(f"Node {self.node_id} at depth {self.node_level} updating max_points to {child_max_points}, max_levels to {child_max_levels}")
#     #
#     #     # Create child nodes for each quadrant, passing the constants from the parent
#     #     self.children = []
#     #
#     #     for boundary in quadrant_boundaries:
#     #         self.root_node.global_count += 1
#     #         child = Quadtree(
#     #             boundary=boundary,
#     #             # points=[],  # Child starts with no points; they'll be re-inserted
#     #             max_points=child_max_points,  # self.max_points Initial value, will be updated by density_func
#     #             max_levels=child_max_levels,  # self.max_levels
#     #             density_func=self.density_func,  # Pass density function to children
#     #             max_levels_func=self.max_levels_func,  # Pass adaptive quadtree level function to children
#     #             node_id=self.root_node.global_count,
#     #             root_node=self.root_node,
#     #             parent=self,
#     #             node_level=self.node_level + 1,
#     #             n_total = self.n_total,  # Pass to children
#     #             alpha = self.alpha,  # Pass parent's alpha
#     #             kappa = self.kappa,  # Pass parent's kappa
#     #             lambda_val = self.lambda_val,  # Pass parent's lambda_val
#     #             min_base = self.min_base,  # Pass parent's min_base
#     #             beta = self.beta,  # Pass parent's beta
#     #             gamma = self.gamma,  # Pass parent's gamma
#     #             delta = self.delta  # Pass parent's delta
#     #         )
#     #         self.children.append(child)
#     #         # Update max depth seen
#     #         if hasattr(self.root_node, 'max_depth'):
#     #             self.root_node.max_depth = max(self.root_node.max_depth, child.node_level)
#     #         else:
#     #             self.root_node.max_depth = child.node_level
#     #         logging.info(
#     #             f"Node {child.node_id} created at current node level {child.node_level}, computed max_levels={child_max_levels}, assigned max_levels={child.max_levels}")
#     #
#     #     # Distribute points to children (Insert all points stored in temp_points into the appropriate child).
#     #     for point in self.temp_points:
#     #         inserted = False
#     #         for child in self.children:
#     #             if child.boundary.contains_point(point.x, point.y):
#     #                 child.insert(point)
#     #                 inserted = True
#     #                 break
#     #         if not inserted:
#     #             logging.warning(
#     #                 f"Point ({point.x}, {point.y}) not inserted into any child node during subdivision of Node {self.node_id}")
#     #
#     #     # Clear temp_points as they have been distributed.
#     #     self.temp_points = []
#     #     self.max_points = self.density_func(self.points, self) if update_max_points else self.max_points
#     #     self.max_levels = self.max_levels_func(self.points, self) if update_max_points else self.max_levels
#
#     def subdivide(self):
#         # Calculate median split points based on points
#         if self.points:
#             lats = [point.y for point in self.points]
#             lons = [point.x for point in self.points]
#             x_mid = pd.Series(lons).median()
#             y_mid = pd.Series(lats).median()
#         else:
#             # Fallback to midpoint if no points
#             x_mid = (self.boundary.x1 + self.boundary.x2) / 2
#             y_mid = (self.boundary.y1 + self.boundary.y2) / 2
#
#         # Define the boundaries for each quadrant
#         quadrant_boundaries = [
#             Rectangle(self.boundary.x1, y_mid, x_mid, self.boundary.y2),  # NW
#             Rectangle(x_mid, y_mid, self.boundary.x2, self.boundary.y2),  # NE
#             Rectangle(self.boundary.x1, self.boundary.y1, x_mid, y_mid),  # SW
#             Rectangle(x_mid, self.boundary.y1, self.boundary.x2, y_mid)  # SE
#         ]
#
#         # Update max_points frequency based on dataset size
#         update_frequency = 3 if self.n_total > 1000000 else 1
#         update_max_points = self.node_level % update_frequency == 0
#
#         child_max_points = self.density_func(self.points, self) if update_max_points else self.max_points
#         child_max_levels = self.max_levels_func(self.points, self) if update_max_points else self.max_levels
#
#         if update_max_points:
#             logging.info(
#                 f"Node {self.node_id} at depth {self.node_level} updating max_points to {child_max_points}, max_levels to {child_max_levels}")
#
#         # Create child nodes for each quadrant
#         self.children = []
#         for boundary in quadrant_boundaries:
#             self.root_node.global_count += 1
#             child = Quadtree(
#                 boundary=boundary,
#                 max_points=child_max_points,
#                 max_levels=child_max_levels,
#                 density_func=self.density_func,
#                 max_levels_func=self.max_levels_func,
#                 node_id=self.root_node.global_count,
#                 root_node=self.root_node,
#                 parent=self,
#                 node_level=self.node_level + 1,
#                 n_total=self.n_total,
#                 alpha=self.alpha,
#                 kappa=self.kappa,
#                 lambda_val=self.lambda_val,
#                 min_base=self.min_base,
#                 beta=self.beta,
#                 gamma=self.gamma,
#                 delta=self.delta
#             )
#             self.children.append(child)
#             if hasattr(self.root_node, 'max_depth'):
#                 self.root_node.max_depth = max(self.root_node.max_depth, child.node_level)
#             else:
#                 self.root_node.max_depth = child.node_level
#             logging.info(
#                 f"Node {child.node_id} created at current node level {child.node_level}, computed max_levels={child_max_levels}, assigned max_levels={child.max_levels}")
#
#         # Distribute points to children
#         for point in self.temp_points:
#             inserted = False
#             for child in self.children:
#                 if child.boundary.contains_point(point.x, point.y):
#                     child.insert(point)
#                     inserted = True
#                     break
#             if not inserted:
#                 logging.warning(
#                     f"Point ({point.x}, {point.y}) not inserted into any child node during subdivision of Node {self.node_id}")
#
#         # Clear temp_points as they have been distributed
#         self.temp_points = []
#         self.max_points = self.density_func(self.points, self) if update_max_points else self.max_points
#         self.max_levels = self.max_levels_func(self.points, self) if update_max_points else self.max_levels
#
#     # Check if the current node is a leaf node (i.e., it has no children).
#     def is_leaf(self):
#         return len(self.children) == 0
#
#
#     # Recursive method to traversal through quadtree using Depth-First Search Algorithm.
#     def traverse_quadtree(self, csv_writer=None, batch_writer=None, batch_timestamp=None):
#         """Recursively log the details of this node and all its children."""
#
#         # Path to store the CSV file
#         output_path = os.path.join(node_dcr, "quadtree_nodes.csv")
#
#         # At the root node, open the file and initialize the batch writer
#         if self.node_level == 0:
#             csvfile = open(output_path, 'w', newline="")
#             csv_writer = csv.writer(csvfile)
#             csv_writer.writerow(["Node_ID", "Points", "Level", "Node_Type", "Timestamp"])
#             batch_writer = []
#             batch_timestamp = time.strftime("%Y-%m-%d %H:%M:%S") + f",{int(time.time() * 1000) % 1000:03d}"
#         else:
#             assert csv_writer is not None, "csv_writer must be provided for non-root nodes"
#             assert batch_writer is not None, "batch_writer must be provided for non-root nodes"
#             assert batch_timestamp is not None, "batch_timestamp must be provided for non-root nodes"
#
#         # Check is current node is Parent or Leaf node.
#         if self.node_id == 0:
#             node_type = "Root_Node"
#         elif self.children:
#             node_type = "Parent_Node"
#         else:
#             node_type = "Leaf_Node"
#         # Collect the current node's details
#         row = [self.node_id, len(self.points), self.node_level, node_type, batch_timestamp]
#         batch_writer.append(row)
#
#         # Write in batches of 100 rows and update timestamp
#         if len(batch_writer) >= 100:
#             csv_writer.writerows(batch_writer)
#             batch_writer.clear()
#             batch_timestamp = time.strftime("%Y-%m-%d %H:%M:%S") + f",{int(time.time() * 1000) % 1000:03d}"
#
#         # Recursively traverse all children
#         for child in self.children:
#             child.traverse_quadtree(csv_writer, batch_writer, batch_timestamp)
#
#         # At the root node, write any remaining rows and close the file
#         if self.node_level == 0:
#             if batch_writer:  # Write any remaining rows
#                 csv_writer.writerows(batch_writer)
#             csvfile.close()
#
#     # Convert a datetime format to Unix timestamp.u,bkh
#     @staticmethod
#     def datetime_to_unix_timestamps(df):
#         # df['Date'] = df['Date'].astype('int64') // 10 ** 9
#         df = df.copy()  # Avoid modifying a view
#         df.loc[:, 'Date'] = pd.to_datetime(df['Date'])  # Ensure Date is in datetime format
#         df.loc[:, 'Date'] = df['Date'].astype('int64') // 10 ** 9  # Convert to Unix timestamp explicitly
#         df.loc[:, 'Date'] = df['Date'].astype('int64')  # Ensure final dtype is int64
#         return df['Date']
#
#     # Scale down the target and predicted values
#     @staticmethod
#     def min_max_scale_values(df, col_name):
#         df = df.copy()  # Prevent modification on a slice
#         # Reshape the column values to a 2D array
#         col_counts = df[col_name].values.reshape(-1,
#                                                  1)  # -1 means NumPy figure out the number of the rows automatically.
#         # 1 means keep a single column.
#         # Initialize the scaler
#         min_max_scaler = MinMaxScaler(feature_range=(100, 105))
#
#         # Fit and transform the column values
#         # df[col_name] = min_max_scaler.fit_transform(col_counts)
#         df.loc[:, col_name] = min_max_scaler.fit_transform(col_counts).astype(int)
#
#         # Return the scaled column (avoiding direct modification of df)
#         return df[col_name]  # Convert back to 1D
#
#
#


import os
import csv
import time
import logging
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json

# Define directories
node_dcr = 'node_dcr'
dcr_dir_csv = 'node_pred_dir_csv'
output_dir_csv = 'output_csv'
model_saved = 'model_saved'

def setup_directories(dir_list):
    """
    Create directories if they do not exist.
    """
    for directory in dir_list:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created directory: {directory}")

# Set up the directories
setup_directories([node_dcr, dcr_dir_csv, output_dir_csv, model_saved])

# Reference dataset size and tuned values
REFERENCE_SIZE = 1140125
TUNED_VALUES = {
    "alpha": 5000,
    "kappa": 100000,
    "lambda_val": 10,
    "min_base": 2000,
    "beta": 10000,
    "gamma": 2,
    "delta": 2
}

def calculate_constants(dataset_size):
    if not isinstance(dataset_size, (int, float)):
        raise TypeError(f"Expected dataset_size to be a number, got {type(dataset_size)}")

    scaling_factor = dataset_size / REFERENCE_SIZE
    new_constants = {}

    for param in ["alpha", "kappa", "min_base", "beta"]:
        new_constants[param] = max(1, round(TUNED_VALUES[param] * scaling_factor))

    log_scale = 1 + math.log(max(1, scaling_factor)) if scaling_factor > 0 else 1
    new_constants["lambda_val"] = min(20, max(5, round(TUNED_VALUES["lambda_val"] * log_scale)))
    new_constants["gamma"] = min(5, max(1, round(TUNED_VALUES["gamma"] * log_scale)))
    new_constants["delta"] = min(5, max(1, round(TUNED_VALUES["delta"] * log_scale)))

    return new_constants

def crime_density(points, self):
    if not points:
        return self.alpha

    crime_counts = [p.Crime_count for p in points]
    variance = pd.Series(crime_counts).var()
    if pd.isna(variance):
        logging.info(f"Node with {len(points)} points has NaN variance, using default max_points={self.alpha}")
        return self.alpha
    logging.info(f"Crime_count variance: {variance}")
    n_points_total = self.n_total
    max_cap = max(self.alpha, min(self.kappa, int(n_points_total / self.lambda_val)))
    base_max = max(self.min_base, min(max_cap, int(self.beta / (1 + variance / self.gamma) + len(points) / self.delta)))
    logging.info(f"Computed max_points: {base_max}, max_cap: {max_cap}")
    return base_max

def adaptive_max_levels(points, self):
    if not points:
        return 5

    n_points_total = self.n_total
    crime_counts = [p.Crime_count for p in points]
    variance = pd.Series(crime_counts).var() if crime_counts else 0.0
    if pd.isna(variance):
        return 5

    eta = 1.5
    mu = int(eta * math.log2(n_points_total)) if n_points_total > 0 else 5
    computed_levels = int(math.log2(n_points_total) + 1 + variance) if n_points_total > 0 else 5
    max_levels = min(mu, computed_levels)
    max_levels = min(15, max(5, max_levels))
    logging.info(f"Computed max_levels: {max_levels}, variance: {variance}, n_points_total: {n_points_total}")
    return max_levels

class InitialQuadtree:
    def __init__(self):
        self.evaluation_results = []

    @staticmethod
    def set_pred_zero(df):
        df = df.copy()
        df.loc[:, 'Date'] = Quadtree.datetime_to_unix_timestamps(df)
        df.loc[:, 'Crime_count'] = Quadtree.min_max_scale_values(df, col_name='Crime_count').round()
        df.loc[:, 'Prediction'] = 0
        return df

    @staticmethod
    def init_quadtree(df, constants):
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

        boundary_rectangle = Rectangle(min(df['Longitude']), min(df['Latitude']), max(df['Longitude']),
                                      max(df['Latitude']))

        quadtree = Quadtree(
            boundary_rectangle,
            density_func=lambda p, self: crime_density(p, self),
            max_levels_func=lambda p, self: adaptive_max_levels(p, self),
            n_total=n_total,
            alpha=constants["alpha"],
            kappa=constants["kappa"],
            lambda_val=constants["lambda_val"],
            min_base=constants["min_base"],
            beta=constants["beta"],
            gamma=constants["gamma"],
            delta=constants["delta"]
        )
        inserted_count = 0
        for point in points:
            if quadtree.insert(point):
                inserted_count += 1
        logging.info(f"Total points inserted: {inserted_count} out of {n_total}")
        if hasattr(quadtree, 'max_depth'):
            logging.info(f"Maximum depth reached: {quadtree.max_depth}")

        # Perform merging of small leaf nodes
        quadtree.merge_small_leaf_nodes(threshold=1000)

        return quadtree

class Point:
    def __init__(self, x, y, index, Date, Time, Hour, Minute, Second, Scl_Longitude, Scl_Latitude,
                 Day_of_Week, Is_Weekend, Day_of_Month, Day_of_Year, Month, Quarter, Year, Week_of_Year,
                 Days_Since_Start, Is_Holiday, Season_Fall, Season_Spring, Season_Summer, Season_Winter,
                 Crime_count, Prediction):
        self.x = x
        self.y = y
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

class Rectangle:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def contains_point(self, x, y):
        return (self.x1 <= x <= self.x2) and (self.y1 <= y <= self.y2)

    def intersects(self, other):
        return not (self.x2 < other.x1 or self.x1 > other.x2 or self.y2 < other.y1 or self.y1 > other.y2)

class Quadtree:
    def __init__(self, boundary, max_points=None, max_levels=None, density_func=None, max_levels_func=None,
                 node_id=0, root_node=None, node_level=0, parent=None, df=None, ex_time=None, n_total=None,
                 alpha=None, kappa=None, lambda_val=None, min_base=None, beta=None, gamma=None, delta=None):
        self.model = None
        self.boundary = boundary
        self.density_func = density_func if density_func is not None else crime_density
        self.max_levels_func = max_levels_func if max_levels_func is not None else adaptive_max_levels
        self.points = []  # Stores actual Point objects
        self.children = []
        self.node_level = node_level
        self.node_id = node_id
        self.parent = parent
        self.df = df
        self.ex_time = ex_time
        self.evaluation_results = []
        self.n_total = n_total
        self.alpha = alpha
        self.kappa = kappa
        self.lambda_val = lambda_val
        self.min_base = min_base
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.merged_pairs = {}  # To store merge mappings

        required_constants = ['alpha', 'kappa', 'lambda_val', 'min_base', 'beta', 'gamma', 'delta']
        for const in required_constants:
            if getattr(self, const) is None:
                raise ValueError(f"Constant '{const}' must be provided and cannot be None")

        self.max_points = max_points if max_points is not None else (
            density_func(self.points, self) if density_func else 1000)
        self.max_levels = max_levels if max_levels is not None else (
            max_levels_func(self.points, self) if max_levels_func else 5)

        if root_node is None:
            self.root_node = self
            self.global_count = 0
        else:
            self.root_node = root_node

        if not isinstance(self.boundary, Rectangle):
            raise ValueError("Boundary must be a Rectangle object")

    def insert(self, point, node_id=None):
        if node_id is None:
            node_id = self.node_id

        if not self.boundary.contains_point(point.x, point.y):
            logging.warning(f"Point ({point.x}, {point.y}) outside boundary of Node {self.node_id}")
            return False

        self.points.append(point)

        if self.is_leaf():
            if len(self.points) >= self.max_points and self.node_level < self.max_levels:
                self.subdivide()
            else:
                if hasattr(self.root_node, 'max_depth'):
                    self.root_node.max_depth = max(self.root_node.max_depth, self.node_level)
                else:
                    self.root_node.max_depth = self.node_level
                return True

        inserted = False
        for child in self.children:
            if child.boundary.contains_point(point.x, point.y):
                inserted = child.insert(point, child.node_id)
                if inserted:
                    break

        if not inserted:
            logging.warning(
                f"Point ({point.x}, {point.y}) not inserted into any child of Node {self.node_id}, keeping in current node")
        return True

    def subdivide(self):
        if self.points:
            lats = [point.y for point in self.points]
            lons = [point.x for point in self.points]
            x_mid = pd.Series(lons).median()
            y_mid = pd.Series(lats).median()
        else:
            x_mid = (self.boundary.x1 + self.boundary.x2) / 2
            y_mid = (self.boundary.y1 + self.boundary.y2) / 2

        # Define the boundaries for each quadrant with non-overlapping regions
        quadrant_boundaries = [
            Rectangle(self.boundary.x1, y_mid, x_mid, self.boundary.y2),  # NW
            Rectangle(x_mid, y_mid, self.boundary.x2, self.boundary.y2),  # NE
            Rectangle(self.boundary.x1, self.boundary.y1, x_mid, y_mid),  # SW
            Rectangle(x_mid, self.boundary.y1, self.boundary.x2, y_mid)  # SE
        ]

        update_frequency = 3 if self.n_total > 1000000 else 1
        update_max_points = self.node_level % update_frequency == 0

        child_max_points = self.density_func(self.points, self) if update_max_points else self.max_points
        child_max_levels = self.max_levels_func(self.points, self) if update_max_points else self.max_levels

        if update_max_points:
            logging.info(
                f"Node {self.node_id} at depth {self.node_level} updating max_points to {child_max_points}, max_levels to {child_max_levels}")

        self.children = []
        for boundary in quadrant_boundaries:
            self.root_node.global_count += 1
            child = Quadtree(
                boundary=boundary,
                max_points=child_max_points,
                max_levels=child_max_levels,
                density_func=self.density_func,
                max_levels_func=self.max_levels_func,
                node_id=self.root_node.global_count,
                root_node=self.root_node,
                parent=self,
                node_level=self.node_level + 1,
                n_total=self.n_total,
                alpha=self.alpha,
                kappa=self.kappa,
                lambda_val=self.lambda_val,
                min_base=self.min_base,
                beta=self.beta,
                gamma=self.gamma,
                delta=self.delta
            )
            self.children.append(child)
            if hasattr(self.root_node, 'max_depth'):
                self.root_node.max_depth = max(self.root_node.max_depth, child.node_level)
            else:
                self.root_node.max_depth = child.node_level
            logging.info(
                f"Node {child.node_id} created at current node level {child.node_level}, computed max_levels={child_max_levels}, assigned max_levels={child.max_levels}")

        # Distribute points to children while retaining them in the parent
        points_to_distribute = self.points.copy()
        # Do NOT clear self.points; retain them for modeling
        for point in points_to_distribute:
            inserted = False
            for child in self.children:
                if child.boundary.contains_point(point.x, point.y):
                    child.insert(point)
                    inserted = True
                    break
            if not inserted:
                # Find the child whose boundary is closest to the point
                closest_child = min(self.children, key=lambda c: min(
                    abs(point.x - c.boundary.x1), abs(point.x - c.boundary.x2),
                    abs(point.y - c.boundary.y1), abs(point.y - c.boundary.y2)
                ))
                # Adjust for numerical precision if on the split line
                if abs(point.x - x_mid) < 1e-10:
                    point.x = x_mid + 1e-10 if point.x >= x_mid else x_mid - 1e-10
                if abs(point.y - y_mid) < 1e-10:
                    point.y = y_mid + 1e-10 if point.y >= y_mid else y_mid - 1e-10
                for child in self.children:
                    if child.boundary.contains_point(point.x, point.y):
                        child.insert(point)
                        inserted = True
                        break
                if not inserted:
                    closest_child.insert(point)
                    logging.warning(
                        f"Point ({point.x}, {point.y}) not inserted into any child node during subdivision of Node {self.node_id}, assigned to closest child Node {closest_child.node_id}")

        self.max_points = self.density_func(self.points, self) if update_max_points else self.max_points
        self.max_levels = self.max_levels_func(self.points, self) if update_max_points else self.max_levels


    # def subdivide(self):
    #     if not self.points:
    #         x_mid = (self.boundary.x1 + self.boundary.x2) / 2
    #         y_mid = (self.boundary.y1 + self.boundary.y2) / 2
    #     else:
    #         # Initial split using median of coordinates
    #         lons = sorted([point.x for point in self.points])
    #         lats = sorted([point.y for point in self.points])
    #         x_mid = lons[len(lons) // 2]
    #         y_mid = lats[len(lats) // 2]
    #
    #         # Function to count points in a region
    #         def count_points_in_region(x1, y1, x2, y2):
    #             return sum(1 for p in self.points if x1 <= p.x < x2 and y1 <= p.y < y2)
    #
    #         # Adjust x_mid to balance points between left and right
    #         target_count = len(self.points) / 2
    #         left_count = count_points_in_region(self.boundary.x1, self.boundary.y1, x_mid, self.boundary.y2)
    #         right_count = len(self.points) - left_count
    #         step = (self.boundary.x2 - self.boundary.x1) / 50  # Smaller initial step for finer adjustments
    #         max_iterations = 200  # Increase iterations for better convergence
    #         iteration = 0
    #         while abs(left_count - right_count) > 0.02 * len(self.points) and iteration < max_iterations:  # Tighter tolerance
    #             if left_count > target_count:
    #                 x_mid -= step
    #             else:
    #                 x_mid += step
    #             left_count = count_points_in_region(self.boundary.x1, self.boundary.y1, x_mid, self.boundary.y2)
    #             right_count = len(self.points) - left_count
    #             step *= 0.95  # Slower step size reduction for more precise adjustments
    #             iteration += 1
    #         if iteration == max_iterations:
    #             logging.warning(f"Node {self.node_id}: X-axis balancing did not converge after {max_iterations} iterations. Final difference: {abs(left_count - right_count)} points.")
    #
    #         # Adjust y_mid to balance points between top and bottom
    #         top_count = count_points_in_region(self.boundary.x1, y_mid, self.boundary.x2, self.boundary.y2)
    #         bottom_count = len(self.points) - top_count
    #         step = (self.boundary.y2 - self.boundary.y1) / 50  # Smaller initial step
    #         iteration = 0
    #         while abs(top_count - bottom_count) > 0.02 * len(self.points) and iteration < max_iterations:  # Tighter tolerance
    #             if top_count > target_count:
    #                 y_mid -= step
    #             else:
    #                 y_mid += step
    #             top_count = count_points_in_region(self.boundary.x1, y_mid, self.boundary.x2, self.boundary.y2)
    #             bottom_count = len(self.points) - top_count
    #             step *= 0.95  # Slower step size reduction
    #             iteration += 1
    #         if iteration == max_iterations:
    #             logging.warning(f"Node {self.node_id}: Y-axis balancing did not converge after {max_iterations} iterations. Final difference: {abs(top_count - bottom_count)} points.")
    #
    #     # Define the boundaries for each quadrant with non-overlapping regions
    #     quadrant_boundaries = [
    #         Rectangle(self.boundary.x1, y_mid, x_mid, self.boundary.y2),  # NW
    #         Rectangle(x_mid, y_mid, self.boundary.x2, self.boundary.y2),  # NE
    #         Rectangle(self.boundary.x1, self.boundary.y1, x_mid, y_mid),  # SW
    #         Rectangle(x_mid, self.boundary.y1, self.boundary.x2, y_mid)  # SE
    #     ]
    #
    #     update_frequency = 3 if self.n_total > 1000000 else 1
    #     update_max_points = self.node_level % update_frequency == 0
    #
    #     child_max_points = self.density_func(self.points, self) if update_max_points else self.max_points
    #     child_max_levels = self.max_levels_func(self.points, self) if update_max_points else self.max_levels
    #
    #     if update_max_points:
    #         logging.info(
    #             f"Node {self.node_id} at depth {self.node_level} updating max_points to {child_max_points}, max_levels to {child_max_levels}")
    #
    #     self.children = []
    #     for boundary in quadrant_boundaries:
    #         self.root_node.global_count += 1
    #         child = Quadtree(
    #             boundary=boundary,
    #             max_points=child_max_points,
    #             max_levels=child_max_levels,
    #             density_func=self.density_func,
    #             max_levels_func=self.max_levels_func,
    #             node_id=self.root_node.global_count,
    #             root_node=self.root_node,
    #             parent=self,
    #             node_level=self.node_level + 1,
    #             n_total=self.n_total,
    #             alpha=self.alpha,
    #             kappa=self.kappa,
    #             lambda_val=self.lambda_val,
    #             min_base=self.min_base,
    #             beta=self.beta,
    #             gamma=self.gamma,
    #             delta=self.delta
    #         )
    #         self.children.append(child)
    #         if hasattr(self.root_node, 'max_depth'):
    #             self.root_node.max_depth = max(self.root_node.max_depth, child.node_level)
    #         else:
    #             self.root_node.max_depth = child.node_level
    #         logging.info(
    #             f"Node {child.node_id} created at current node level {child.node_level}, computed max_levels={child_max_levels}, assigned max_levels={child.max_levels}")
    #
    #     # Distribute points to children while retaining them in the parent
    #     points_to_distribute = self.points.copy()
    #     for point in points_to_distribute:
    #         inserted = False
    #         for child in self.children:
    #             if child.boundary.contains_point(point.x, point.y):
    #                 child.insert(point)
    #                 inserted = True
    #                 break
    #         if not inserted:
    #             closest_child = min(self.children, key=lambda c: min(
    #                 abs(point.x - c.boundary.x1), abs(point.x - c.boundary.x2),
    #                 abs(point.y - c.boundary.y1), abs(point.y - c.boundary.y2)
    #             ))
    #             if abs(point.x - x_mid) < 1e-10:
    #                 point.x = x_mid + 1e-10 if point.x >= x_mid else x_mid - 1e-10
    #             if abs(point.y - y_mid) < 1e-10:
    #                 point.y = y_mid + 1e-10 if point.y >= y_mid else y_mid - 1e-10
    #             for child in self.children:
    #                 if child.boundary.contains_point(point.x, point.y):
    #                     child.insert(point)
    #                     inserted = True
    #                     break
    #             if not inserted:
    #                 closest_child.insert(point)
    #                 logging.warning(
    #                     f"Point ({point.x}, {point.y}) not inserted into any child node during subdivision of Node {self.node_id}, assigned to closest child Node {closest_child.node_id}")
    #
    #     # Check for children with 0 points
    #     for child in self.children:
    #         if len(child.points) == 0:
    #             logging.warning(f"Node {child.node_id} has 0 points after subdivision of Node {self.node_id}.")
    #
    #     self.max_points = self.density_func(self.points, self) if update_max_points else self.max_points
    #     self.max_levels = self.max_levels_func(self.points, self) if update_max_points else self.max_levels

    def is_leaf(self):
        return len(self.children) == 0

    # New method: Traverse the quadtree to collect leaf nodes
    def get_leaf_nodes(self, leaf_nodes=None):
        if leaf_nodes is None:
            leaf_nodes = []
        if self.is_leaf():
            leaf_nodes.append(self)
        else:
            for child in self.children:
                child.get_leaf_nodes(leaf_nodes)
        return leaf_nodes

    # Merge small leaf nodes
    def merge_small_leaf_nodes(self, threshold=1000):
        merged_pairs = {}
        iteration = 0

        while True:
            iteration += 1
            leaf_nodes = self.get_leaf_nodes()
            small_leaves = [
                node for node in leaf_nodes
                if 0 < len(node.points) < threshold and  # Exclude nodes with 0 points
                node.node_id not in merged_pairs and
                node.node_id not in merged_pairs.values()
            ]

            if not small_leaves:
                print(f"Merging complete after {iteration} iterations. No small leaf nodes remain.")
                break

            print(f"Iteration {iteration}: Found {len(small_leaves)} leaf nodes with fewer than {threshold} points.")

            merges_in_iteration = 0
            for small_node in small_leaves:
                small_node_id = small_node.node_id
                parent = small_node.parent

                if parent is None:
                    continue

                siblings = [
                    child for child in parent.children
                    if child.is_leaf() and child.node_id != small_node_id and
                    child.node_id not in merged_pairs and child.node_id not in merged_pairs.values()
                ]

                if not siblings:
                    continue

                target_sibling = min(siblings, key=lambda x: len(x.points))
                target_node_id = target_sibling.node_id

                original_small_points = len(small_node.points)
                target_sibling.points.extend(small_node.points)
                small_node.points = []
                parent.children.remove(small_node)

                merged_pairs[small_node_id] = target_node_id
                print(f"Merging Node {small_node_id} ({original_small_points} points) into Node {target_node_id} (now {len(target_sibling.points)} points)")
                merges_in_iteration += 1

            if merges_in_iteration == 0:
                print(f"Iteration {iteration}: No merges possible. {len(small_leaves)} small leaf nodes remain unmerged.")
                break

        self.merged_pairs = merged_pairs
        with open("node_dcr/merged_pairs.json", "w") as f:
            json.dump(merged_pairs, f)
        print(f"Saved merge mapping to node_dcr/merged_pairs.json: {merged_pairs}")

    # New method: Get points for a node, including merged nodes
    def get_points_for_node(self, node, all_points=None):
        points = node.points

        merged_node_id = None
        if node.node_id in self.merged_pairs:
            merged_node_id = self.merged_pairs[node.node_id]
        else:
            for small_id, target_id in self.merged_pairs.items():
                if node.node_id == target_id:
                    merged_node_id = small_id
                    break

        if merged_node_id is not None:
            merged_node = self.get_node_by_id(merged_node_id)
            if merged_node:
                points = points + merged_node.points

        return points

    # New method: Get a node by its ID
    def get_node_by_id(self, node_id, node=None):
        if node is None:
            node = self
        if node.node_id == node_id:
            return node
        for child in node.children:
            result = self.get_node_by_id(node_id, child)
            if result is not None:
                return result
        return None

    # New method: Train on quadtree with merged nodes
    def train_on_quadtree(self):
        leaf_nodes = self.get_leaf_nodes()

        for node in leaf_nodes:
            node_id = node.node_id

            if node_id in self.merged_pairs:
                continue

            node_points = self.get_points_for_node(node)
            num_points = len(node_points)
            print(f"Training on Node {node_id} with {num_points} points")

            if num_points < 1000:
                print(f"Warning: Node {node_id} still has {num_points} points after merging.")

            # Placeholder for training
            # model = train_model(node_points)
            # save_model(model, node_id)

    # New method: Get combined boundaries for a node
    def get_combined_boundaries(self, node):
        min_lon, max_lon, min_lat, max_lat = (self.boundary.x1, self.boundary.x2, self.boundary.y1, self.boundary.y2)

        merged_node_id = None
        if node.node_id in self.merged_pairs:
            merged_node_id = self.merged_pairs[node.node_id]
        else:
            for small_id, target_id in self.merged_pairs.items():
                if node.node_id == target_id:
                    merged_node_id = small_id
                    break

        if merged_node_id is not None:
            merged_node = self.get_node_by_id(merged_node_id)
            if merged_node:
                m_min_lon, m_max_lon, m_min_lat, m_max_lat = (merged_node.boundary.x1, merged_node.boundary.x2,
                                                              merged_node.boundary.y1, merged_node.boundary.y2)
                min_lon = min(min_lon, m_min_lon)
                max_lon = max(max_lon, m_max_lon)
                min_lat = min(min_lat, m_min_lat)
                max_lat = max(max_lat, m_max_lat)

        return min_lon, max_lon, min_lat, max_lat

    # New method: Visualize the quadtree after merging
    def visualize_quadtree(self):
        leaf_nodes = self.get_leaf_nodes()

        fig, ax = plt.subplots(figsize=(10, 10))

        for node in leaf_nodes:
            node_id = node.node_id

            if node_id in self.merged_pairs:
                continue

            min_lon, max_lon, min_lat, max_lat = self.get_combined_boundaries(node)

            width = max_lon - min_lon
            height = max_lat - min_lat
            rect = patches.Rectangle(
                (min_lon, min_lat), width, height,
                linewidth=1, edgecolor="black", facecolor="none", alpha=0.5
            )
            ax.add_patch(rect)

            node_points = len(node.points)
            if node_id in self.merged_pairs.values():
                for small_id, target_id in self.merged_pairs.items():
                    if target_id == node_id:
                        small_node = self.get_node_by_id(small_id)
                        node_points += len(small_node.points)
            ax.text(
                (min_lon + max_lon) / 2, (min_lat + max_lat) / 2,
                f"Node {node_id}\n{node_points} pts",
                ha="center", va="center", fontsize=8
            )

        ax.set_xlim(self.boundary.x1, self.boundary.x2)
        ax.set_ylim(self.boundary.y1, self.boundary.y2)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Quadtree Visualization After Merging Small Leaf Nodes")

        plt.savefig("node_dcr/quadtree_visualization_after_merging.pdf")
        plt.close()
        print("Saved quadtree visualization to node_dcr/quadtree_visualization_after_merging.pdf")

    # New method: Verify merges
    def verify_merges(self):
        leaf_nodes = self.get_leaf_nodes()
        effective_leaf_nodes = set()

        for node in leaf_nodes:
            node_id = node.node_id
            if node_id not in self.merged_pairs:
                effective_leaf_nodes.add(node_id)

        print(f"Effective number of leaf nodes after merging: {len(effective_leaf_nodes)}")

        for node_id in effective_leaf_nodes:
            node = self.get_node_by_id(node_id)
            node_points = len(node.points)
            for small_id, target_id in self.merged_pairs.items():
                if target_id == node_id:
                    small_node = self.get_node_by_id(small_id)
                    node_points += len(small_node.points)
            if node_points < 1000:
                print(f"Warning: Effective Node {node_id} has {node_points} points (below threshold).")
            else:
                print(f"Effective Node {node_id} has {node_points} points.")

    # Modified method: Remove reliance on quadtree_nodes.csv
    def traverse_quadtree(self, csv_writer=None, batch_writer=None, batch_timestamp=None):
        """Recursively log the details of this node and all its children for debugging purposes."""
        output_path = os.path.join(node_dcr, "quadtree_nodes.csv")

        if self.node_level == 0:
            csvfile = open(output_path, 'w', newline="")
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Node_ID", "Points", "Level", "Node_Type", "Min_Longitude", "Max_Longitude", "Min_Latitude", "Max_Latitude", "Timestamp"])
            batch_writer = []
            batch_timestamp = time.strftime("%Y-%m-%d %H:%M:%S") + f",{int(time.time() * 1000) % 1000:03d}"
        else:
            assert csv_writer is not None, "csv_writer must be provided for non-root nodes"
            assert batch_writer is not None, "batch_writer must be provided for non-root nodes"
            assert batch_timestamp is not None, "batch_timestamp must be provided for non-root nodes"

        if self.node_id == 0:
            node_type = "Root_Node"
        elif self.children:
            node_type = "Parent_Node"
        else:
            node_type = "Leaf_Node"

        row = [
            self.node_id, len(self.points), self.node_level, node_type,
            self.boundary.x1, self.boundary.x2, self.boundary.y1, self.boundary.y2,
            batch_timestamp
        ]
        batch_writer.append(row)

        if len(batch_writer) >= 100:
            csv_writer.writerows(batch_writer)
            batch_writer.clear()
            batch_timestamp = time.strftime("%Y-%m-%d %H:%M:%S") + f",{int(time.time() * 1000) % 1000:03d}"

        for child in self.children:
            child.traverse_quadtree(csv_writer, batch_writer, batch_timestamp)

        if self.node_level == 0:
            if batch_writer:
                csv_writer.writerows(batch_writer)
            csvfile.close()

    @staticmethod
    def datetime_to_unix_timestamps(df):
        df = df.copy()
        df.loc[:, 'Date'] = pd.to_datetime(df['Date'])
        df.loc[:, 'Date'] = df['Date'].astype('int64') // 10 ** 9
        df.loc[:, 'Date'] = df['Date'].astype('int64')
        return df['Date']

    @staticmethod
    def min_max_scale_values(df, col_name):
        df = df.copy()
        col_counts = df[col_name].values.reshape(-1, 1)
        min_max_scaler = MinMaxScaler(feature_range=(100, 105))
        df.loc[:, col_name] = min_max_scaler.fit_transform(col_counts).astype(int)
        return df[col_name]