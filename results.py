import pandas as pd
import os
from visualise import Visualise as vis

# Define the path to the CSV file
node_dcr = "node_dcr"  # Adjust this if your directory path is different
csv_path = os.path.join(node_dcr, "quadtree_nodes.csv")

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_path)
print("First few rows of the DataFrame:")
print(df.head())

points_by_level = vis.point_distribution_by_level(df)

vis.node_count_by_level(df)

leaf_nodes = vis.leaf_node_analysis(df)

vis.points_vs_level(df, points_by_level, leaf_nodes)

vis.point_distribution_validation(df)
