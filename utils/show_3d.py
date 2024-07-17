import plotly.graph_objects as go
import matplotlib.pyplot as plt


class Show_In_3D:
    """
    A class to visualize 3D points and optional trajectory using Plotly and Matplotlib.

    Attributes
    ----------
    points_3d : np.ndarray
        An array of 3D points.
    trajectory : np.ndarray, optional
        An array of 3D points representing a trajectory.

    Methods
    -------
    show_3d():
        Displays the 3D points and optional trajectory using Plotly and Matplotlib.
    """

    def __init__(self, points_3d, trajectory=None):
        """
        Initializes the Show_In_3D class with 3D points and an optional trajectory.

        Parameters
        ----------
        points_3d : np.ndarray
            An array of 3D points.
        trajectory : np.ndarray, optional
            An array of 3D points representing a trajectory.
        """
        self.points_3d = points_3d
        self.trajectory = trajectory
        self.x_3d = self.points_3d[:, 0]
        self.y_3d = self.points_3d[:, 1]
        self.z_3d = self.points_3d[:, 2]
        self.show_trajectory = trajectory is not None
        if self.show_trajectory:
            self.x_traj = trajectory[:, 0]
            self.y_traj = trajectory[:, 1]
            self.z_traj = trajectory[:, 2]

    def show_3d(self):
        """
        Displays the 3D points and optional trajectory using Plotly and Matplotlib.
        """
        plots = []

        # Create a 3D scatter plot for the points using Plotly
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
            # Create a 3D line plot for the trajectory using Plotly
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

        # Combine the plots in a single figure
        fig = go.Figure(data=plots)

        # Set the layout of the Plotly figure
        fig.update_layout(
            scene=dict(
                xaxis_title='X Axis',
                yaxis_title='Y Axis',
                zaxis_title='Z Axis'
            ),
            title='3D Points and Trajectory Visualization'
        )

        # Show the Plotly figure
        fig.show()

        # Additionally, show the same data using Matplotlib
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.x_3d, self.y_3d, self.z_3d, marker='.', s=5, label='3D Points')

        if self.show_trajectory:
            ax.plot(self.x_traj, self.y_traj, self.z_traj, color='red', label='Trajectory')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend()
        plt.show()
