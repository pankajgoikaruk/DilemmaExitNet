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
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

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
    "alpha": 2000, # 5000
    "kappa": 50000, # 100000
    "lambda_val": 10,
    "min_base": 5000, # 2000
    "beta": 50000, # 10000
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
            Crime_count=row['Crime_count'], Prediction=row['Prediction'], Crime_count_lag1=row['Crime_count_lag1'],
            Crime_count_lag2=row['Crime_count_lag2'], Crime_count_lag3=row['Crime_count_lag3'],
            Crime_count_roll_mean_7d=row['Crime_count_roll_mean_7d'],
            Hour_sin=row['Hour_sin'], Hour_cos=row['Hour_cos'],
            Month_sin=row['Month_sin'], Month_cos=row['Month_cos']
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

        return quadtree

class Point:
    def __init__(self, x, y, index, Date, Time, Hour, Minute, Second, Scl_Longitude, Scl_Latitude,
                 Day_of_Week, Is_Weekend, Day_of_Month, Day_of_Year, Month, Quarter, Year,
                 Week_of_Year, Days_Since_Start, Is_Holiday, Season_Fall, Season_Spring,
                 Season_Summer, Season_Winter, Crime_count, Prediction, Crime_count_lag1=0,
                 Crime_count_lag2=0, Crime_count_lag3=0, Crime_count_roll_mean_7d=0,
                 Hour_sin=0, Hour_cos=0, Month_sin=0, Month_cos=0):
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
        self.Crime_count_lag1 = Crime_count_lag1
        self.Crime_count_lag2 = Crime_count_lag2
        self.Crime_count_lag3 = Crime_count_lag3
        self.Crime_count_roll_mean_7d = Crime_count_roll_mean_7d
        self.Hour_sin = Hour_sin
        self.Hour_cos = Hour_cos
        self.Month_sin = Month_sin
        self.Month_cos = Month_cos

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

    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)

class Quadtree:
    def __init__(self, boundary, max_points=None, max_levels=None, density_func=None, max_levels_func=None,
                 node_id=0, root_node=None, node_level=0, parent=None, df=None, ex_time=None, n_total=None,
                 alpha=None, kappa=None, lambda_val=None, min_base=None, beta=None, gamma=None, delta=None):
        self.model = None # To hold trained model
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
        self.is_merged = False  # To track if the node was merged

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

    def is_leaf(self):
        return len(self.children) == 0

    # Traverse the quadtree to collect leaf nodes
    def get_leaf_nodes(self, leaf_nodes=None):
        if leaf_nodes is None:
            leaf_nodes = []
        if self.is_leaf():
            leaf_nodes.append(self)
        else:
            for child in self.children:
                child.get_leaf_nodes(leaf_nodes)
        return leaf_nodes

    def compute_density_percentiles(self):
        # Save current state to CSV to compute densities
        self.traverse_quadtree()
        df = pd.read_csv("node_dcr/quadtree_nodes.csv")
        densities = df["Density"].values
        q25 = pd.Series(densities).quantile(0.25)
        q75 = pd.Series(densities).quantile(0.75)
        return q25, q75

    # Updated merge_small_leaf_nodes method with density-based merging
    def merge_small_leaf_nodes(self, threshold=5000, density_outlier_factor=1.5):
        merged_pairs = {}
        iteration = 0
        max_combined_threshold = threshold * 2.5  # Allow merging if combined points are below this limit

        # Compute density percentiles to identify outliers
        q25, q75 = self.compute_density_percentiles()
        iqr = q75 - q25
        density_lower_bound = q25 - density_outlier_factor * iqr
        density_upper_bound = q75 + density_outlier_factor * iqr

        while True:
            iteration += 1
            leaf_nodes = self.get_leaf_nodes()

            # Identify small nodes (based on point count) and density outliers
            small_leaves = []
            density_outlier_nodes = []

            for node in leaf_nodes:
                # Skip nodes that have already been merged
                if node.node_id in merged_pairs or node.node_id in merged_pairs.values():
                    continue

                # Compute node density
                node_area = (node.boundary.x2 - node.boundary.x1) * (node.boundary.y2 - node.boundary.y1)
                node_density = len(node.points) / node_area if node_area > 0 else 0  # Fix: Use len(node.points)

                # Check for small nodes (based on point count)
                if 0 < len(node.points) < threshold:
                    small_leaves.append((node, len(node.points)))

                # Check for density outliers
                if node_density < density_lower_bound or node_density > density_upper_bound:
                    density_outlier_nodes.append((node, node_density))

            # Sort nodes: small nodes by point count (ascending), density outliers by density (low to high)
            small_leaves.sort(key=lambda x: x[1])
            density_outlier_nodes.sort(key=lambda x: x[1])

            # Combine lists, prioritizing small nodes
            nodes_to_merge = small_leaves + [(node, density) for node, density in density_outlier_nodes if
                                             node not in [n[0] for n in small_leaves]]

            if not nodes_to_merge:
                print(f"Merging complete after {iteration} iterations. No small leaf nodes or density outliers remain.")
                break

            print(
                f"Iteration {iteration}: Found {len(nodes_to_merge)} nodes to consider for merging (small nodes: {len(small_leaves)}, density outliers: {len(density_outlier_nodes)}).")

            merges_in_iteration = 0
            updated_nodes = set()  # Track nodes updated in this iteration

            for node_to_merge, _ in nodes_to_merge:
                # Skip if the node was already updated in this iteration
                if node_to_merge.node_id in updated_nodes:
                    continue

                node_to_merge_id = node_to_merge.node_id
                parent = node_to_merge.parent

                if parent is None:
                    continue

                # Get siblings that are leaf nodes and not yet merged
                siblings = [
                    child for child in parent.children
                    if child.is_leaf() and child.node_id != node_to_merge_id and
                       child.node_id not in merged_pairs and child.node_id not in merged_pairs.values()
                ]

                if not siblings:
                    continue

                # Compute density of the node to merge
                node_to_merge_area = (node_to_merge.boundary.x2 - node_to_merge.boundary.x1) * (
                            node_to_merge.boundary.y2 - node_to_merge.boundary.y1)
                node_to_merge_density = len(
                    node_to_merge.points) / node_to_merge_area if node_to_merge_area > 0 else 0  # Fix: Use len(node_to_merge.points)

                # Find eligible siblings based on combined point count
                eligible_siblings = [
                    sibling for sibling in siblings
                    if (len(node_to_merge.points) + len(sibling.points)) <= max_combined_threshold
                ]

                if not eligible_siblings:
                    continue

                # Choose the sibling with the closest density
                sibling_densities = [
                    (sibling, len(sibling.points) / ((sibling.boundary.x2 - sibling.boundary.x1) * (
                                sibling.boundary.y2 - sibling.boundary.y1)))
                    for sibling in eligible_siblings
                    if (sibling.boundary.x2 - sibling.boundary.x1) * (sibling.boundary.y2 - sibling.boundary.y1) > 0
                    # Avoid division by zero
                ]
                sibling_densities.sort(key=lambda x: abs(x[1] - node_to_merge_density))
                target_sibling = sibling_densities[0][0] if sibling_densities else None

                if not target_sibling:
                    continue

                target_node_id = target_sibling.node_id

                # Perform the merge
                original_points = len(node_to_merge.points)
                target_sibling.points.extend(node_to_merge.points)
                node_to_merge.points = []
                parent.children.remove(node_to_merge)

                # Set is_merged flags
                node_to_merge.is_merged = True
                target_sibling.is_merged = True

                merged_pairs[node_to_merge_id] = target_node_id
                print(
                    f"Merging Node {node_to_merge_id} ({original_points} points) into Node {target_node_id} (now {len(target_sibling.points)} points)")
                merges_in_iteration += 1

                # Add both nodes to updated_nodes to prevent further merges in this iteration
                updated_nodes.add(node_to_merge_id)
                updated_nodes.add(target_node_id)

            if merges_in_iteration == 0:
                print(f"Iteration {iteration}: No merges possible. {len(nodes_to_merge)} nodes remain unmerged.")
                break

        self.merged_pairs = merged_pairs
        with open("node_dcr/merged_pairs.json", "w") as f:
            json.dump(merged_pairs, f)
        print(f"Saved merge mapping to node_dcr/merged_pairs.json: {merged_pairs}")

    # # Merge small leaf nodes
    # def merge_small_leaf_nodes(self, threshold=5000): # 1000
    #     merged_pairs = {}
    #     iteration = 0
    #     max_combined_threshold = threshold * 2.5  # Allow merging if combined points are below this limit
    #
    #     while True:
    #         iteration += 1
    #         leaf_nodes = self.get_leaf_nodes()
    #
    #         # Only consider nodes that haven't been merged yet and have points below the threshold
    #         small_leaves = [
    #             node for node in leaf_nodes
    #             if 0 < len(node.points) < threshold and  # Exclude nodes with 0 points
    #             node.node_id not in merged_pairs and
    #             node.node_id not in merged_pairs.values()
    #         ]
    #
    #         if not small_leaves:
    #             print(f"Merging complete after {iteration} iterations. No small leaf nodes remain.")
    #             break
    #
    #         print(f"Iteration {iteration}: Found {len(small_leaves)} leaf nodes with fewer than {threshold} points.")
    #
    #         merges_in_iteration = 0
    #         # Create a set to track nodes that have been updated in this iteration
    #         updated_nodes = set()
    #
    #         for small_node in small_leaves:
    #             # Skip if the small node was already updated in this iteration
    #             if small_node.node_id in updated_nodes:
    #                 continue
    #
    #             small_node_id = small_node.node_id
    #             parent = small_node.parent
    #
    #             if parent is None:
    #                 continue
    #
    #             # Get siblings that are leaf nodes and not yet merged
    #             siblings = [
    #                 child for child in parent.children
    #                 if child.is_leaf() and child.node_id != small_node_id and
    #                    child.node_id not in merged_pairs and child.node_id not in merged_pairs.values()
    #             ]
    #
    #             if not siblings:
    #                 continue
    #
    #             # Find the sibling with the fewest points, but allow merging if combined points are below max_combined_threshold
    #             eligible_siblings = [
    #                 sibling for sibling in siblings
    #                 if (len(small_node.points) + len(sibling.points)) <= max_combined_threshold
    #             ]
    #
    #             if not eligible_siblings:
    #                 continue
    #
    #             target_sibling = min(eligible_siblings, key=lambda x: len(x.points))
    #             target_node_id = target_sibling.node_id
    #
    #             # Perform the merge
    #             original_small_points = len(small_node.points)
    #             target_sibling.points.extend(small_node.points)
    #             small_node.points = []
    #             parent.children.remove(small_node)
    #
    #             # Set is_merged flags
    #             small_node.is_merged = True
    #             target_sibling.is_merged = True
    #
    #             merged_pairs[small_node_id] = target_node_id
    #             print(f"Merging Node {small_node_id} ({original_small_points} points) into Node {target_node_id} (now {len(target_sibling.points)} points)")
    #             merges_in_iteration += 1
    #
    #             # Add both nodes to updated_nodes to prevent further merges in this iteration
    #             updated_nodes.add(small_node_id)
    #             updated_nodes.add(target_node_id)
    #
    #         if merges_in_iteration == 0:
    #             print(f"Iteration {iteration}: No merges possible. {len(small_leaves)} small leaf nodes remain unmerged.")
    #             break
    #
    #     self.merged_pairs = merged_pairs
    #     with open("node_dcr/merged_pairs.json", "w") as f:
    #         json.dump(merged_pairs, f)
    #     print(f"Saved merge mapping to node_dcr/merged_pairs.json: {merged_pairs}")

    # Get points for a node, including merged nodes
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

    # Get a node by its ID
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

    def train_on_quadtree(self):
        leaf_nodes = self.get_leaf_nodes()
        feature_columns = [
            'Scl_Longitude', 'Scl_Latitude', 'Date', 'Hour', 'Day_of_Week', 'Is_Weekend',
            'Day_of_Month', 'Day_of_Year', 'Month', 'Quarter', 'Year', 'Week_of_Year',
            'Days_Since_Start', 'Is_Holiday', 'Season_Fall', 'Season_Spring', 'Season_Summer', 'Season_Winter',
            'Crime_count_lag1', 'Crime_count_lag2', 'Crime_count_lag3', 'Crime_count_roll_mean_7d',
            'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos'
        ]
        target_column = 'Crime_count'

        for node in leaf_nodes:
            node_id = node.node_id
            if node_id in self.merged_pairs:
                continue

            # Get points with merged nodes
            node_points = self.get_points_for_node(node)
            num_points = len(node_points)
            print(f"Training Node {node_id} ({num_points} points)")

            if num_points < 1000:
                print(f"Warning: Node {node_id} has {num_points} points, skipping training.")
                continue

            # Convert points to DataFrame
            data = {
                col: [getattr(pt, col) for pt in node_points]
                for col in feature_columns + [target_column, 'index']
            }
            df = pd.DataFrame(data)

            # Sort by Date for consistency
            df = df.sort_values(by='Date')

            # Prepare features and target (use full data for training)
            X = df[feature_columns]
            y = df[target_column]

            # Train XGBoost model on full node data
            start_time = time.time()
            model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X, y)
            end_time = time.time()
            node.ex_time = end_time - start_time

            # Compute feature importance
            importance = model.feature_importances_
            feature_importance = dict(zip(feature_columns, importance))
            print(f"Feature Importance for Node {node_id}: {feature_importance}")

            # Store model
            node.model = model
            model_path = os.path.join('model_saved', f"node_{node_id}.pkl")
            joblib.dump(model, model_path)
            print(f"Saved model: {model_path}")

            # Save predictions on training data
            df['Prediction'] = model.predict(X)
            pred_df = df[['index', 'Scl_Longitude', 'Scl_Latitude', target_column, 'Prediction']]
            pred_path = os.path.join('node_pred_dir_csv', f"node_{node_id}_pred.csv")
            pred_df.to_csv(pred_path, index=False)
            print(f"Saved predictions: {pred_path}")

            # Store training time
            node.evaluation_results.append({
                'Node_ID': node_id,
                'Node_Level': node.node_level,
                'Points': num_points,
                'Ex_Time': node.ex_time,
                'Feature_Importance': feature_importance
            })

    def evaluate_on_validation(self, val_df):
        leaf_nodes = self.get_leaf_nodes()
        feature_columns = [
            'Scl_Longitude', 'Scl_Latitude', 'Date', 'Hour', 'Day_of_Week', 'Is_Weekend',
            'Day_of_Month', 'Day_of_Year', 'Month', 'Quarter', 'Year', 'Week_of_Year',
            'Days_Since_Start', 'Is_Holiday', 'Season_Fall', 'Season_Spring', 'Season_Summer', 'Season_Winter',
            'Crime_count_lag1', 'Crime_count_lag2', 'Crime_count_lag3', 'Crime_count_roll_mean_7d',
            'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos'
        ]
        target_column = 'Crime_count'

        # Prepare validation data
        val_df = val_df.copy()
        val_df = val_df.sort_values(by='Date')
        # val_df['Crime_count_lag1'] = val_df[target_column].shift(1).fillna(val_df[target_column].mean())

        # Assign each validation point to a leaf node
        val_points = [
            Point(
                x=row['Longitude'], y=row['Latitude'], index=row['index'], Date=row['Date'], Time=row.get('Time', 0),
                Hour=row['Hour'], Minute=row.get('Minute', 0), Second=row.get('Second', 0),
                Scl_Longitude=row['Scl_Longitude'], Scl_Latitude=row['Scl_Latitude'], Day_of_Week=row['Day_of_Week'],
                Is_Weekend=row['Is_Weekend'], Day_of_Month=row['Day_of_Month'], Day_of_Year=row['Day_of_Year'],
                Month=row['Month'], Quarter=row['Quarter'], Year=row['Year'], Week_of_Year=row['Week_of_Year'],
                Days_Since_Start=row['Days_Since_Start'], Is_Holiday=row['Is_Holiday'], Season_Fall=row['Season_Fall'],
                Season_Spring=row['Season_Spring'], Season_Summer=row['Season_Summer'],
                Season_Winter=row['Season_Winter'],
                Crime_count=row['Crime_count'], Prediction=0, Crime_count_lag1=row['Crime_count_lag1'],
                Crime_count_lag2=row['Crime_count_lag2'], Crime_count_lag3=row['Crime_count_lag3'],
                Crime_count_roll_mean_7d=row['Crime_count_roll_mean_7d'],
                Hour_sin=row['Hour_sin'], Hour_cos=row['Hour_cos'],
                Month_sin=row['Month_sin'], Month_cos=row['Month_cos']
            ) for _, row in val_df.iterrows()
        ]

        # Map validation points to leaf nodes
        node_val_points = {node.node_id: [] for node in leaf_nodes}
        for point in val_points:
            node = self.find_leaf_node(point)
            if node and node.node_id not in self.merged_pairs:
                node_val_points[node.node_id].append(point)

        # Evaluate each node on its validation points
        for node in leaf_nodes:
            node_id = node.node_id
            if node_id in self.merged_pairs or not node.model:
                continue

            val_node_points = node_val_points.get(node_id, [])
            if not val_node_points:
                print(f"Warning: Node {node_id} has no validation points, skipping evaluation.")
                continue

            # Convert validation points to DataFrame
            data = {
                col: [getattr(pt, col) for pt in val_node_points]
                for col in feature_columns + [target_column, 'index']
            }
            val_node_df = pd.DataFrame(data)

            X_val = val_node_df[feature_columns]
            y_val = val_node_df[target_column]

            # Predict using the node's model
            y_pred = node.model.predict(X_val)

            # Compute metrics
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            n = len(y_val)
            p = len(feature_columns)
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else 0
            mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
            smape = np.mean(2 * np.abs(y_pred - y_val) / (np.abs(y_val) + np.abs(y_pred))) * 100

            print(f"Validation Node {node_id} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}, AdjR2: {adj_r2:.2f}, MAPE: {mape:.2f}%, SMAPE: {smape:.2f}%")

            # Update evaluation results with validation metrics
            node.evaluation_results[0].update({
                'Val_Points': len(val_node_points),
                'Val_RMSE': rmse,
                'Val_MAE': mae,
                'Val_R2': r2,
                'Val_AdjR2': adj_r2,
                'Val_MAPE': mape,
                'Val_SMAPE': smape
            })

        # Aggregate and save evaluation results
        eval_df = pd.DataFrame([res for node in leaf_nodes for res in node.evaluation_results])
        if not eval_df.empty:
            avg_rmse = eval_df['Val_RMSE'].mean()
            avg_mae = eval_df['Val_MAE'].mean()
            avg_r2 = eval_df['Val_R2'].mean()
            avg_adj_r2 = eval_df['Val_AdjR2'].mean()
            avg_mape = eval_df['Val_MAPE'].mean()
            avg_smape = eval_df['Val_SMAPE'].mean()
            avg_ex_time = eval_df['Ex_Time'].mean()
            print(f"Average Validation RMSE: {avg_rmse:.2f}, MAE: {avg_mae:.2f}, R2: {avg_r2:.2f}, AdjR2: {avg_adj_r2:.2f}, MAPE: {avg_mape:.2f}%, SMAPE: {avg_smape:.2f}%, Avg Ex_Time: {avg_ex_time:.2f}s")
            eval_df.to_csv('output_csv/quadtree_model_eval.csv', index=False)
            print("Saved evaluation results: output_csv/quadtree_model_eval.csv")

    def find_leaf_node(self, point):
        node = self
        while node.children:
            for child in node.children:
                if child.boundary.contains_point(point.x, point.y):
                    node = child
                    break
            else:
                break
        return node if not node.children else None


    # Get combined boundaries for a node
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

    # Visualize the quadtree after merging
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

    # Verify merges
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
            if node_points < 5000: # 1000
                print(f"Warning: Effective Node {node_id} has {node_points} points (below threshold).")
            else:
                print(f"Effective Node {node_id} has {node_points} points.")

    # Quadtree Traversal Method:
    def traverse_quadtree(self, csv_writer=None, batch_writer=None, batch_timestamp=None):
        """Recursively log the details of this node and all its children for debugging purposes."""
        output_path = os.path.join(node_dcr, "quadtree_nodes.csv")

        if self.node_level == 0:
            csvfile = open(output_path, 'w', newline="")
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Node_ID", "Points", "Level", "Node_Type", "Min_Longitude", "Max_Longitude", "Min_Latitude", "Max_Latitude", "Timestamp", "Parent_ID", "Is_Merged", "Density"])
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

        # Compute density (points per unit area)
        area = self.boundary.area()
        density = len(self.points) / area if area > 0 else 0

        # Get parent_id (-1 for root node)
        parent_id = -1 if self.parent is None else self.parent.node_id

        row = [
            self.node_id,
            len(self.points),
            self.node_level,
            node_type,
            self.boundary.x1,
            self.boundary.x2,
            self.boundary.y1,
            self.boundary.y2,
            batch_timestamp,
            parent_id,
            int(self.is_merged),
            density
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

    # Get maximum depth
    def get_max_depth(self):
        return self.max_depth

    # Get total number of nodes
    def get_total_nodes(self):
        def count_nodes(node):
            count = 1  # Count the current node
            for child in node.children:
                count += count_nodes(child)
            return count

        return count_nodes(self)

    # Range query to find points within a rectangle
    def range_query(self, rect):
        points = []
        self._range_query(rect, points)
        return points

    def _range_query(self, rect, points, node=None):
        if node is None:
            node = self
        if not rect.intersects(node.boundary):
            return
        if node.is_leaf():
            for point in node.points:
                if rect.contains_point(point.x, point.y):
                    points.append(point)
        else:
            for child in node.children:
                self._range_query(rect, points, child)

    # Helper method to estimate memory usage (approximate)
    def estimate_memory_usage(self):
        # Approximate memory usage:
        # - Each node: ~200 bytes (rough estimate for object overhead, boundaries, etc.)
        # - Each point: ~100 bytes (rough estimate for Point object with attributes)
        total_nodes = self.get_total_nodes()
        total_points = sum(len(node.points) for node in self.get_leaf_nodes())
        node_memory = total_nodes * 200  # bytes
        point_memory = total_points * 100  # bytes
        total_memory = (node_memory + point_memory) / (1024 * 1024)  # Convert to MB
        return total_memory

    # Evaluation method for the quadtree
    def evaluate_quadtree(self, output_file="node_dcr/quadtree_evaluation.txt"):
        evaluation_results = []

        # 1. Structural Metrics
        max_depth = self.get_max_depth()
        total_nodes = self.get_total_nodes()
        leaf_nodes = self.get_leaf_nodes()
        num_leaf_nodes = len(leaf_nodes)

        # Average points per leaf node
        points_per_leaf = [len(node.points) for node in leaf_nodes]
        avg_points_per_leaf = np.mean(points_per_leaf) if points_per_leaf else 0
        variance_points_per_leaf = np.var(points_per_leaf) if points_per_leaf else 0

        # Merged nodes analysis
        df = pd.read_csv("node_dcr/quadtree_nodes.csv")
        num_merged_nodes = len(df[df["Is_Merged"] == 1])
        merged_points = df[df["Is_Merged"] == 1]["Points"].sum()

        # 2. Performance Metrics: Range Query
        # Define a sample range query rectangle (e.g., a small central region)
        mid_lon = (self.boundary.x1 + self.boundary.x2) / 2
        mid_lat = (self.boundary.y1 + self.boundary.y2) / 2
        range_rect = Rectangle(
            mid_lon - 0.01, mid_lat - 0.01,
            mid_lon + 0.01, mid_lat + 0.01
        )

        start_time = time.time()
        range_points = self.range_query(range_rect)
        range_query_time = time.time() - start_time
        num_points_found = len(range_points)

        # 3. Memory Usage
        memory_usage_mb = self.estimate_memory_usage()

        # 4. Density Distribution
        density_stats = df["Density"].describe().to_dict()

        # Compile results
        evaluation_results.append(f"Evaluation for Median-Based Quadtree")
        evaluation_results.append(f"=====================================")
        evaluation_results.append(f"Structural Metrics:")
        evaluation_results.append(f"- Maximum Depth: {max_depth}")
        evaluation_results.append(f"- Total Number of Nodes: {total_nodes}")
        evaluation_results.append(f"- Number of Leaf Nodes: {num_leaf_nodes}")
        evaluation_results.append(f"- Average Points per Leaf Node: {avg_points_per_leaf:.2f}")
        evaluation_results.append(f"- Variance of Points per Leaf Node: {variance_points_per_leaf:.2f}")
        evaluation_results.append(f"- Number of Merged Nodes: {num_merged_nodes}")
        evaluation_results.append(f"- Total Points in Merged Nodes: {merged_points}")
        evaluation_results.append(f"\nPerformance Metrics:")
        evaluation_results.append(
            f"- Range Query Time (for central 0.02x0.02 degree region): {range_query_time:.4f} seconds")
        evaluation_results.append(f"- Number of Points Found in Range Query: {num_points_found}")
        evaluation_results.append(f"- Estimated Memory Usage: {memory_usage_mb:.2f} MB")
        evaluation_results.append(f"\nDensity Distribution:")
        for key, value in density_stats.items():
            evaluation_results.append(f"- {key}: {value:.2f}")

        # Save to file
        with open(output_file, "w") as f:
            f.write("\n".join(evaluation_results))
        print(f"Evaluation results saved to {output_file}")

        return {
            "max_depth": max_depth,
            "total_nodes": total_nodes,
            "num_leaf_nodes": num_leaf_nodes,
            "avg_points_per_leaf": avg_points_per_leaf,
            "variance_points_per_leaf": variance_points_per_leaf,
            "num_merged_nodes": num_merged_nodes,
            "merged_points": merged_points,
            "range_query_time": range_query_time,
            "num_points_found": num_points_found,
            "memory_usage_mb": memory_usage_mb,
            "density_stats": density_stats
        }

# # Add the new methods to the Quadtree class
# Quadtree.estimate_memory_usage = estimate_memory_usage
# Quadtree.evaluate_quadtree = evaluate_quadtree

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