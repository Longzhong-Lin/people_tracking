#!/usr/bin/env python

import rospy
import numpy as np
import tf
from tf.transformations import euler_from_quaternion, quaternion_matrix
from scipy.ndimage import rotate
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import OccupancyGrid
from matplotlib import pyplot as plt

class GridMapToOccupancy:
    def __init__(self):
        rospy.init_node('grid_map_to_occupancy_converter', anonymous=True)
        
        self.fig, self.ax = plt.subplots()
        self.im = None
        self.cbar = None
        self.occupancy_grid = None
        
        self.occupancy_grid_publisher = rospy.Publisher('/local_map', OccupancyGrid, queue_size=10)
        rospy.Subscriber('/elevation_mapping/elevation_map_filter', GridMap, self.grid_map_callback)

        self.tf_listener = tf.TransformListener()

    def grid_map_callback(self, msg):
        try:
            # Get the transformation between the frames
            (trans, rot) = self.tf_listener.lookupTransform('body', 'cam_odom', rospy.Time(0))
            transform_matrix = quaternion_matrix(rot)
            
            # Convert GridMap to OccupancyGrid first
            occupancy_grid = self.convert_to_occupancy_grid(msg)
            
            # Then apply the rotation and flipping based on the transformation
            if occupancy_grid:
                rotated_and_flipped_data = self.apply_transform(occupancy_grid, transform_matrix)
                occupancy_grid.data = rotated_and_flipped_data.astype(np.int8).tolist()
                self.occupancy_grid_publisher.publish(occupancy_grid)
                self.occupancy_grid = occupancy_grid
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.loginfo("TF exception: %s", e)

    def convert_to_occupancy_grid(self, grid_map):
        resolution = grid_map.info.resolution
        width_cells = int(grid_map.info.length_x / resolution)
        height_cells = int(grid_map.info.length_y / resolution)

        # Create an empty OccupancyGrid
        occupancy_grid = OccupancyGrid()
        occupancy_grid.header = grid_map.info.header
        occupancy_grid.header.frame_id = 'base_link'
        occupancy_grid.info.resolution = resolution
        occupancy_grid.info.width = width_cells
        occupancy_grid.info.height = height_cells
        occupancy_grid.info.origin.position.x = -grid_map.info.length_x / 2
        occupancy_grid.info.origin.position.y = -grid_map.info.length_y / 2

        # Extract the layer and reshape
        smooth_layer_index = grid_map.layers.index("smooth")
        smooth_data = np.array(grid_map.data[smooth_layer_index].data)
        occupancy_values = smooth_data.reshape((height_cells, width_cells))

        # Map the height values to occupancy
        center_height = occupancy_values[height_cells//2, width_cells//2]
        height_tolerance = 0.1
        occupancy_grid.data = (np.abs(occupancy_values - center_height) > height_tolerance).astype(int) * 100

        return occupancy_grid

    def apply_transform(self, occupancy_grid, transform_matrix):
        data = np.array(occupancy_grid.data).reshape((occupancy_grid.info.height, occupancy_grid.info.width))
        angle = np.degrees(np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])) - 90
        rotated_data = rotate(data, angle, reshape=False, mode='nearest')
        rotated_data = np.flip(rotated_data, axis=0)
        rotated_data = np.flip(rotated_data, axis=1)
        return rotated_data.flatten()

    def visualize(self, occupancy_grid):
        data = np.array(occupancy_grid.data).reshape((occupancy_grid.info.height, occupancy_grid.info.width))
        if self.im is None:
            self.im = self.ax.imshow(data, cmap='gray', origin='lower')
            self.cbar = self.fig.colorbar(self.im, ax=self.ax, label='Occupancy Probability')
            self.ax.set_title('Occupancy Grid Map')
        else:
            self.im.set_data(data)
            self.im.set_clim(0, 100)
        plt.draw()

    def run(self, visualize=False):
        if visualize:
            plt.ion()
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            if visualize and self.occupancy_grid is not None:
                self.visualize(self.occupancy_grid)
                plt.pause(0.01)
            rate.sleep()

if __name__ == '__main__':
    visualizer = GridMapToOccupancy()
    visualizer.run()
