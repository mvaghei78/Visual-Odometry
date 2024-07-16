import plotly.graph_objects as go
import matplotlib.pyplot as plt

class Show_In_3D:
    def __init__(self, points_3d, trajectory=None):
        self.points_3d = points_3d
        self.trajectory = trajectory
        self.x_3d = self.points_3d[:, 0]
        self.y_3d = self.points_3d[:, 1]
        self.z_3d = self.points_3d[:, 2]
        self.show_trajectory = False
        if not(trajectory is None):
            self.show_trajectory = True
            self.x_traj = trajectory[:, 0]
            self.y_traj = trajectory[:, 1]
            self.z_traj = trajectory[:, 2]

    def show_3d(self):
        plots = []
        # Create a 3D scatter plot for the points
        points_plot = go.Scatter3d(
            x=self.x_3d, y=self.y_3d, z=self.z_3d,
            mode='markers',
            marker=dict(
                size=5,
                color=self.z_3d,  # Set color to the z values
                colorscale='Viridis',  # Choose a colorscale
                opacity=0.8
            ),
            name='3D Points'
        )
        plots.append(points_plot)
        if self.show_trajectory:
            # Create a 3D line plot for the trajectory
            trajectory_plot = go.Scatter3d(
                x=self.x_traj, y=self.y_traj, z=self.z_traj,
                mode='lines',
                line=dict(
                    color='red',
                    width=2
                ),
                name='Trajectory'
            )
            plots.append(trajectory_plot)

        # Combine the two plots
        fig = go.Figure(data=plots)

        # Set plot layout
        fig.update_layout(
            scene=dict(
                xaxis_title='X Axis',
                yaxis_title='Y Axis',
                zaxis_title='Z Axis'
            ),
            title='3D Points and Trajectory Visualization'
        )

        # Show plot
        fig.show()

        # Additionally, show the same data using Matplotlib
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.x_3d, self.y_3d, self.z_3d, marker='.', s=5)
        if self.show_trajectory:
            ax.plot(self.x_traj, self.y_traj, self.z_traj, color='red')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

# import plotly.graph_objects as go
# import matplotlib.pyplot as plt
#
# class Show_In_3D:
#     def __init__(self, points_3d):
#         self.points_3d = points_3d
#         self.x = self.points_3d[:, 0]
#         self.y = self.points_3d[:, 1]
#         self.z = self.points_3d[:, 2]
#
#     def show_3d(self):
#         # Create a 3D scatter plot
#         fig = go.Figure(data=[go.Scatter3d(
#             x=self.x, y=self.y, z=self.z,
#             mode='markers',
#             marker=dict(
#                 size=5,
#                 color=self.z,  # Set color to the z values
#                 colorscale='Viridis',  # Choose a colorscale
#                 opacity=0.8
#             )
#         )])
#
#         # Set plot layout
#         fig.update_layout(
#             scene=dict(
#                 xaxis_title='X Axis',
#                 yaxis_title='Y Axis',
#                 zaxis_title='Z Axis'
#             ),
#             title='3D Points Visualization'
#         )
#
#         # Show plot
#         fig.show()
#
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.scatter(self.x, self.y, self.z, marker='.', s=5)
#
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')
#         plt.show()
