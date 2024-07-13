import plotly.graph_objects as go
import matplotlib.pyplot as plt

class Show_In_3D:
    def __init__(self, points_3d):
        self.points_3d = points_3d
        self.x = self.points_3d[:, 0]
        self.y = self.points_3d[:, 1]
        self.z = self.points_3d[:, 2]

    def show_3d(self):
        # Create a 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=self.x, y=self.y, z=self.z,
            mode='markers',
            marker=dict(
                size=5,
                color=self.z,  # Set color to the z values
                colorscale='Viridis',  # Choose a colorscale
                opacity=0.8
            )
        )])

        # Set plot layout
        fig.update_layout(
            scene=dict(
                xaxis_title='X Axis',
                yaxis_title='Y Axis',
                zaxis_title='Z Axis'
            ),
            title='3D Points Visualization'
        )

        # Show plot
        fig.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.x, self.y, self.z, marker='.', s=5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
