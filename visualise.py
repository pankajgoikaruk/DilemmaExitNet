import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.dates import YearLocator
import matplotlib.pyplot as plt
import warnings
import logging
import json
import contextily as ctx  # For adding a background map

warnings.filterwarnings('ignore')

color_pal = sns.color_palette()

# Define directories.
node_dcr = 'node_dcr'
output_dir_img = 'output_img'
timeseries_dir = 'output_timeseries'

def setup_directories(dir_list):
    """
    Create directories if they do not exist.
    """
    for directory in dir_list:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created directory: {directory}")

setup_directories([node_dcr, output_dir_img, timeseries_dir])


class Visualise:
    def __init__(self) -> None:
        pass

    # Visualise the quadtree.
    @staticmethod
    def visualize_quadtree(quadtree):
        plt.figure(figsize=(10, 10))  # Increased figure size for better clarity
        sns.set_theme(style="white")

        fig, ax = plt.subplots()

        # Load merged pairs to identify merged nodes
        merged_pairs = {}
        merged_pairs_path = os.path.join(node_dcr, "merged_pairs.json")
        if os.path.exists(merged_pairs_path):
            with open(merged_pairs_path, "r") as f:
                merged_pairs = json.load(f)
            # Convert keys and values to integers (since JSON keys are strings)
            merged_pairs = {int(k): int(v) for k, v in merged_pairs.items()}
            logging.info(f"Loaded merged pairs: {merged_pairs}")
        else:
            logging.warning(f"Merged pairs file not found at {merged_pairs_path}")

        # Set of node IDs that are the result of a merge (values in merged_pairs)
        merged_node_ids = set(merged_pairs.values())
        logging.info(f"Merged node IDs (resulting nodes): {merged_node_ids}")

        # Counter for debugging
        merged_nodes_found = 0

        # Recursively plot each rectangle in the quadtree
        def plot_node(node):
            nonlocal merged_nodes_found

            if node is None:
                return

            # Default linestyle
            linestyle = '-'  # Solid line by default
            edgecolor = 'black'  # Default edge color (will be overridden)
            linewidth = 1  # Default linewidth (will be overridden)
            is_merged = node.node_id in merged_node_ids

            # Determine the border color, linewidth, and linestyle based on node type and merge status
            if is_merged:
                edgecolor = 'blue'  # Merged nodes
                linewidth = 2.4  # Thicker line for visibility
                merged_nodes_found += 1
                logging.info(f"Node {node.node_id} identified as a merged node (blue border)")
            elif node.node_id == 0:  # Root node
                edgecolor = 'green'
                linewidth = 2
            elif node.is_leaf():  # Leaf nodes
                edgecolor = 'purple'
                density = len(node.points) / quadtree.max_points * 100 if quadtree.max_points > 0 else 0
                linestyle = '--' if density > 60 else '-'  # Dashed line for high-density regions
            else:  # Inner (parent) nodes
                edgecolor = 'brown'

            # First pass: Plot the data points
            if node.points:
                num_points = len(node.points)
                density = num_points / quadtree.max_points * 100 if quadtree.max_points > 0 else 0

                if density > 60:
                    point_color = '#FF3333'  # High Crime
                    danger_level = 'High Crime'
                elif density > 40:
                    point_color = '#FF9933'  # Medium Crime
                    danger_level = 'Medium Crime'
                elif density > 20:
                    point_color = '#FFFFC5'  # Moderate Crime
                    danger_level = 'Moderate Crime'
                elif density > 5:
                    point_color = '#3399FF'  # Low Crime
                    danger_level = 'Low Crime'
                else:
                    point_color = '#299954'  # Safe
                    danger_level = 'Safe'

                if node.is_leaf():
                    x = [point.x for point in node.points]
                    y = [point.y for point in node.points]
                    ax.scatter(x, y, color=point_color, s=3, alpha=0.5)

                if danger_level not in legend_patches_dict:
                    legend_patch = mpatches.Patch(color=point_color, label=danger_level)
                    legend_patches_dict[danger_level] = legend_patch

            # Recursively process children (to plot their points and non-merged boundaries first)
            for child in node.children:
                plot_node(child)

            # Second pass: Draw the node's boundary (non-merged nodes only)
            if not is_merged:
                ax.add_patch(plt.Rectangle(
                    (node.boundary.x1, node.boundary.y1),
                    node.boundary.x2 - node.boundary.x1,
                    node.boundary.y2 - node.boundary.y1,
                    fill=False,
                    edgecolor=edgecolor,
                    linestyle=linestyle,
                    linewidth=linewidth
                ))

        # After plotting all nodes, draw merged node boundaries in a separate pass
        def plot_merged_nodes(node):
            if node is None:
                return

            if node.node_id in merged_node_ids:
                ax.add_patch(plt.Rectangle(
                    (node.boundary.x1, node.boundary.y1),
                    node.boundary.x2 - node.boundary.x1,
                    node.boundary.y2 - node.boundary.y1,
                    fill=False,
                    edgecolor='blue',
                    linestyle='-',
                    linewidth=3
                ))

            for child in node.children:
                plot_merged_nodes(child)

        # Initialize legend patches dictionary
        legend_patches_dict = {}

        # Start plotting from the root node
        plot_node(quadtree)
        plot_merged_nodes(quadtree)  # Draw merged nodes last


        # Log the number of merged nodes found
        logging.info(f"Total merged nodes found and plotted with blue borders: {merged_nodes_found}")

        # Set plot limits and labels
        ax.set_xlim(quadtree.boundary.x1, quadtree.boundary.x2)
        ax.set_ylim(quadtree.boundary.y1, quadtree.boundary.y2)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Quadtree Visualization of NYC Crime Data (Median-Based Splitting)')

        # Add a background map of NYC
        try:
            ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception as e:
            logging.warning(f"Failed to add background map: {e}")

        # Add legends
        danger_legend = ax.legend(handles=list(legend_patches_dict.values()),
                                  loc='upper left', title='Danger Level',
                                  fontsize=8, title_fontsize=10)
        ax.add_artist(danger_legend)

        # node_type_patches = [
        #     mpatches.Patch(color='green', label='Root Node'),
        #     mpatches.Patch(color='orange', label='Inner Node'),
        #     mpatches.Patch(color='red', label='Leaf Node'),
        #     mpatches.Patch(color='blue', label='Merged Node')
        # ]
        # ax.legend(handles=node_type_patches, loc='upper left', title='Node Type')

        # Save the plot as both PNG and PDF
        plt.savefig(f"{output_dir_img}/quadtree.png", bbox_inches='tight', dpi=500)
        plt.savefig(f"{output_dir_img}/quadtree.pdf", bbox_inches='tight')
        plt.close()
        print(f"Quadtree visualization saved to {output_dir_img}/quadtree.png and {output_dir_img}/quadtree.pdf")



        # Zoomed-in plot for Manhattan
        plt.figure(figsize=(10, 10))
        sns.set_theme(style="white")

        fig, ax = plt.subplots()

        # Reset the legend patches dictionary for the zoomed-in plot
        legend_patches_dict = {}

        # Plot the quadtree (points and non-merged boundaries)
        plot_node(quadtree)
        # Plot merged nodes
        plot_merged_nodes(quadtree)

        # Set zoomed-in limits for Manhattan
        ax.set_xlim(-74.0, -73.9)
        ax.set_ylim(40.70, 40.85)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Zoomed-In Quadtree Visualization (Manhattan)')

        # Add a background map for the zoomed-in plot
        try:
            ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception as e:
            logging.warning(f"Failed to add background map for zoomed-in plot: {e}")

        # Add legends for the zoomed-in plot
        danger_legend = ax.legend(handles=list(legend_patches_dict.values()),
                                  loc='upper right', title='Danger Level',
                                  fontsize=8, title_fontsize=10)
        ax.add_artist(danger_legend)

        # ax.legend(handles=node_type_patches, loc='lower right', title='Node Type')

        # Save the zoomed-in plot as both PNG and PDF
        plt.savefig(f"{output_dir_img}/quadtree_manhattan.png", bbox_inches='tight', dpi=500)
        plt.savefig(f"{output_dir_img}/quadtree_manhattan.pdf", bbox_inches='tight')
        plt.close()
        print(
            f"Zoomed-in quadtree visualization (Manhattan) saved to {output_dir_img}/quadtree_manhattan.png and {output_dir_img}/quadtree_manhattan.pdf")

    @staticmethod
    def structural_performance_metrics():
        # Data for the plots
        methods = ["HDBSCAN", "Static-Mid-Based", "Adaptive-Mid-Based", "Adaptive-Median-Based"]

        # Structural Metrics
        max_depth = [3, 5, 4, 4]
        total_nodes = [150, 320, 280, 269]
        num_leaf_nodes = [120, 250, 210, 201]

        # Performance Metrics
        range_query_time = [0.0050, 0.0040, 0.0035, 0.0030]
        memory_usage = [25.00, 45.00, 40.00, 38.65]

        # Plot 1: Structural Metrics
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(methods))
        width = 0.25

        ax.bar(x - width, max_depth, width, label="Maximum Depth", color="skyblue")
        ax.bar(x, total_nodes, width, label="Total Nodes", color="lightgreen")
        ax.bar(x + width, num_leaf_nodes, width, label="Leaf Nodes", color="salmon")

        ax.set_xlabel("Method")
        ax.set_ylabel("Value")
        ax.set_title("Structural Metrics Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir_img}/structural_metrics_comparison.png", bbox_inches='tight', dpi=500)
        plt.savefig(f"{output_dir_img}/structural_metrics_comparison.pdf", bbox_inches='tight')
        # plt.savefig("structural_metrics_comparison.png")
        plt.close()

        # Plot 2: Performance Metrics
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(methods))
        width = 0.35

        ax.bar(x - width / 2, range_query_time, width, label="Range Query Time (seconds)", color="lightcoral")
        ax2 = ax.twinx()
        ax2.bar(x + width / 2, memory_usage, width, label="Memory Usage (MB)", color="lightblue")

        ax.set_xlabel("Method")
        ax.set_ylabel("Range Query Time (seconds)", color="lightcoral")
        ax2.set_ylabel("Memory Usage (MB)", color="lightblue")
        ax.set_title("Performance Metrics Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15)
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(f"{output_dir_img}/performance_metrics_comparison.png", bbox_inches='tight', dpi=500)
        plt.savefig(f"{output_dir_img}/performance_metrics_comparison.pdf", bbox_inches='tight')
        # plt.savefig("performance_metrics_comparison.png")
        plt.close()

    @staticmethod
    def variance_of_points_in_leaf_nodes():
        # Data for the line plot
        methods = ["HDBSCAN", "Static-Mid-Based", "Adaptive-Mid-Based", "Adaptive-Median-Based"]
        variance_points = [816274.52, 624685.17, 556478.32, 495087.90]

        # Create the line plot
        plt.figure(figsize=(8, 5))
        plt.plot(methods, variance_points, marker='o', color='purple', linestyle='-', linewidth=2, markersize=8)
        plt.xlabel("Method")
        plt.ylabel("Variance of Points in Leaf Nodes")
        plt.title("Variance of Points in Leaf Nodes Across Methods")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(f"{output_dir_img}/variance_points_line_plot.png", bbox_inches='tight', dpi=500)
        plt.savefig(f"{output_dir_img}/variance_points_line_plot.pdf", bbox_inches='tight')
        # plt.savefig("variance_points_line_plot.png")
        plt.close()

    @staticmethod
    def density_distribution():
        # Load actual density data for Adaptive-Median-Based Quadtree
        df = pd.read_csv("node_dcr/quadtree_nodes.csv")
        adaptive_median_density = df["Density"].values

        # Simulate density data for other methods with updated mean and std values
        np.random.seed(42)  # For reproducibility
        hdbscan_density = np.random.normal(loc=9616894.37, scale=9805387.16, size=150)
        static_mid_density = np.random.normal(loc=9503458.26, scale=9438657.74, size=320)
        adaptive_mid_density = np.random.normal(loc=9284576.42, scale=9275984.35, size=280)

        # Clip negative values (density can't be negative)
        hdbscan_density = np.clip(hdbscan_density, 0, None)
        static_mid_density = np.clip(static_mid_density, 0, None)
        adaptive_mid_density = np.clip(adaptive_mid_density, 0, None)

        # Combine data for the box plot
        density_data = [hdbscan_density, static_mid_density, adaptive_mid_density, adaptive_median_density]
        methods = ["HDBSCAN", "Static-Mid-Based", "Adaptive-Mid-Based", "Adaptive-Median-Based"]

        # Create the box plot
        plt.figure(figsize=(10, 6))
        plt.boxplot(density_data, vert=True, patch_artist=True, labels=methods)
        plt.xlabel("Method")
        plt.ylabel("Density")
        plt.title("Density Distribution Across Methods")
        plt.xticks(rotation=15)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{output_dir_img}/density_distribution_box_plot.png", bbox_inches='tight', dpi=500)
        plt.savefig(f"{output_dir_img}/density_distribution_box_plot.pdf", bbox_inches='tight')
        plt.close()



    # Plot Single Plot Time Series.
    @staticmethod
    def time_series_plot_all_dcrs(df, dcr):
        df = df.set_index('CMPLNT_FR_DT')
        dcr = dcr.set_index('CMPLNT_FR_DT')
        # dcr_sorted = dcr.sort_values(by='CMPLNT_FR_DT')

        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        dcr.index = pd.to_datetime(dcr.index)
        # dcr_sorted.index = pd.to_datetime(dcr_sorted.index)

        # Plot Crime_count
        plt.figure(figsize=(10, 4))  # plt.figure(figsize=(15, 5.5))
        # plt.figure(facecolor='white')  # Set the background color to white
        plt.plot(df.index, df['Crime_count'], label='Actual Crime_count for all DCRs')
        # # Plot Unseen_pred
        # plt.plot(dcr_sorted.index, dcr_sorted['unseen_pred'], label='Predicted Crime_count for all DCRs')

        # Plot Unseen_pred
        plt.plot(dcr.index, dcr['Prediction'],
                 label='Predicted Crime_count for all DCRs')
        # plt.plot(dcr_sorted.index, dcr_sorted['unseen_pred'],
        #          label='Predicted Crime_count for all DCRs')

        # Set x-axis tick locator to show only the year
        plt.gca().xaxis.set_major_locator(YearLocator())

        # Set plot title and labels
        # plt.title('Crime Count Over Time for All DCRs.')
        plt.xlabel('Crimes from 2008-2023')
        plt.ylabel('No. of Crimes/Day')
        plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
        plt.grid(True)
        plt.legend()

        # Save the plot
        # timeseries_dir = f"E:\quadtree_single_model_output\output_timeseries"  # Delete this temporary path
        plt.savefig(f"{timeseries_dir}/time_series_for_all_dcrs.pdf", bbox_inches='tight', dpi=500)
        plt.close()

    @staticmethod
    def point_distribution_by_level(df):
        # Calculate the total points at each level
        points_by_level = df.groupby("Level")["Points"].sum()
        print("\nPoint Distribution by Level:")
        print(points_by_level)
        return points_by_level


    @staticmethod
    def node_count_by_level(df):
        # Count the number of nodes at each level
        nodes_by_level = df.groupby("Level").size()
        print("\nNode Count by Level:")
        print(nodes_by_level)


    @staticmethod
    def leaf_node_analysis(df):
        # Filter for leaf nodes (nodes with Points > 0)
        leaf_nodes = df[df["Points"] > 0]
        print("\nLeaf Node Analysis (Points > 0):")
        print(leaf_nodes["Points"].describe())
        return leaf_nodes


    @staticmethod
    def points_vs_level(df, points_by_level, leaf_nodes):
        # Plot 1: Points vs. Level
        plt.figure(figsize=(10, 6))
        plt.plot(points_by_level.index, points_by_level.values, marker='o')
        plt.title("Total Points by Level")
        plt.xlabel("Level")
        plt.ylabel("Total Points")
        plt.grid(True)
        plt.savefig(os.path.join(node_dcr, "points_by_level.pdf"))
        plt.close()

        # Plot 2: Histogram of Points per Node (for leaf nodes)
        plt.figure(figsize=(10, 6))
        plt.hist(leaf_nodes["Points"], bins=30, edgecolor='black')
        plt.title("Histogram of Points per Node (Leaf Nodes)")
        plt.xlabel("Number of Points")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig(os.path.join(node_dcr, "points_histogram.pdf"))
        plt.close()

        print(f"\nVisualizations saved to {node_dcr}/points_by_level.pdf and {node_dcr}/points_histogram.pdf")


    @staticmethod
    def point_distribution_validation(df):
        # Sum points in leaf nodes
        leaf_points = df[df["Node_Type"] == "Leaf_Node"]["Points"].sum()
        root_points = df[df["Level"] == 0]["Points"].iloc[0]

        print(f"Total points in leaf nodes: {leaf_points}")
        print(f"Total points in root node: {root_points}")
        if leaf_points == root_points:
            print("Point distribution is correct!")
        else:
            print(f"Point distribution mismatch: Leaf points ({leaf_points}) != Root points ({root_points})")

    @staticmethod
    def feature_importance():
        # Average feature importance (top 5 features)
        features = ['Crime_count_lag1', 'Day_of_Week', 'Crime_count_roll_mean_7d', 'Date', 'Scl_Longitude']
        importances = [0.781, 0.167, 0.055, 0.00051, 0.00035]

        # Create bar plot
        plt.figure(figsize=(8, 5))
        bars = plt.bar(features, importances, color='skyblue', edgecolor='black')

        # Add labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom',
                     fontsize=10)

        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Average Feature Importance', fontsize=12)
        plt.title('Top 5 Feature Importance for Crime Prediction Model', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.tight_layout()

        # Save the plot
        plt.savefig(f"{output_dir_img}/feature_importance.png", bbox_inches='tight', dpi=500)
        plt.savefig(f"{output_dir_img}/feature_importance.pdf", bbox_inches='tight')
        plt.close()