import numpy as np

class Generate_Output:
    """
    A class to generate .ply files for visualizing point clouds.

    Methods
    -------
    generate_ply: Generates a .ply file from the given point cloud and colors.
    """

    def generate_ply(self, path: str, point_cloud: np.ndarray, colors: np.ndarray) -> None:
        """
        Generates the .ply file which can be used to open the point cloud.

        Parameters
        ----------
        path : str
            The path where the .ply file will be saved.
        point_cloud : np.ndarray
            The point cloud data as a numpy array.
        colors : np.ndarray
            The color data for each point in the point cloud.
        """
        # Reshape the point cloud and scale the points
        out_points = point_cloud.reshape(-1, 3) * 200
        out_colors = colors.reshape(-1, 3)
        print(out_colors.shape, out_points.shape)

        # Combine the points and colors into a single array
        verts = np.hstack([out_points, out_colors])

        # Center the points by subtracting the mean
        mean = np.mean(verts[:, :3], axis=0)
        scaled_verts = verts[:, :3] - mean

        # Calculate distances from the mean and filter points
        dist = np.sqrt(scaled_verts[:, 0] ** 2 + scaled_verts[:, 1] ** 2 + scaled_verts[:, 2] ** 2)
        indx = np.where(dist < np.mean(dist) + 300)
        verts = verts[indx]

        # Define the PLY file header
        ply_header = '''ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            property uchar blue
            property uchar green
            property uchar red
            end_header
            '''

        # Write the PLY file
        with open(path, 'w') as f:
            f.write(ply_header % dict(vert_num=len(verts)))
            np.savetxt(f, verts, '%f %f %f %d %d %d')
