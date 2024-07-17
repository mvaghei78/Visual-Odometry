import os
import glob
import matplotlib.pyplot as plt
import cv2
from natsort import natsorted


class DatasetProcess:
    """
    A class for processing datasets, including reading image frames from a folder and extracting frames from a video.

    Methods
    -------
    read_frames: Reads and returns a list of image frames from the specified folder, sorted in numerical order.
    extract_frames: Extracts frames from a video at specified intervals and saves them to the output folder with specified naming format.
    """

    def read_frames(self, images_folder: str) -> list:
        """
        Reads and returns a list of image frames from the specified folder, sorted in numerical order.

        Parameters
        ----------
        :param str images_folder: The folder containing the image frames to read.

        Returns
        -------
        :return: A list of image frames read from the folder.
        :rtype list
        """
        pth = os.path.join(images_folder, '*.jpg')
        image_paths = glob.glob(pth)
        # Sort the paths using natsort to ensure numerical order
        sorted_image_paths = natsorted(image_paths)
        # for f_path in sorted_image_paths:
        #     print(f_path)
        return [plt.imread(f_path) for f_path in sorted_image_paths]

    def extract_frames(self, video_path: str, interval: int, output_folder: str, name_format: str) -> None:
        """
        Extracts frames from a video at specified intervals and saves them to the output folder with specified naming format.

        Parameters
        ----------
        :param str video_path: The path to the video file from which frames are to be extracted.
        :param int interval: The interval at which frames are extracted (every 'interval' frames).
        :param str output_folder: The folder where the extracted frames will be saved.
        :param name_format: The format for naming the saved frames, with a placeholder for the frame number.
        """
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Open the video file
        video_capture = cv2.VideoCapture(video_path)

        # Check if video opened successfully
        if not video_capture.isOpened():
            raise ValueError(f"Error opening video file: {video_path}")

        # Get video properties
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Iterate over the video frames
        frame_number = 0
        save_frame_number = 1

        while True:
            # Set the position of the next frame to capture
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            # Read the frame
            success, frame = video_capture.read()

            if not success:
                break  # Exit loop if no more frames

            # Save the frame if it's in the interval
            if frame_number % interval == 0:
                # Generate the filename
                filename = name_format % save_frame_number
                file_path = os.path.join(output_folder, filename)

                # Save the frame
                cv2.imwrite(file_path, frame)
                print(f'frame {file_path} saved')
                # Increment the save frame number
                save_frame_number += 1

            # Increment the frame number
            frame_number += 1

        # Release the video capture object
        video_capture.release()

        print(f"Frames extracted and saved to {output_folder}")


if __name__ == '__main__':
    dp = DatasetProcess()
    dp.extract_frames('../dataset/h108_90.MP4', 10, '../dataset/drone_high_altitude/', 'drone_img_high_altitude_%d.jpg')
