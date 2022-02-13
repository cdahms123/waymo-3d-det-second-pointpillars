# 0_visualize_dataset.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    # prevent TensorFlow from using the GPU

import numpy as np
import math
import plotly.graph_objects as PlotlyGraphObjects
from typing import List
import pprint

import tensorflow as tf
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils, box_utils
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils

TRAIN_DATASET_LOC = '/home/cdahms/WaymoPerceptionDataset/training'
FILE_LOC = 'segment-616184888931414205_2020_000_2040_000_with_camera_labels.tfrecord'

SHOW_PLOTLY_MOUSEOVERS = False

def main():
    tripData = tf.data.TFRecordDataset(os.path.join(TRAIN_DATASET_LOC, FILE_LOC), compression_type='')

    print('\n' + 'type(tripData): ')
    print(type(tripData))

    for frameIdx, frameData in enumerate(tripData):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(frameData.numpy()))

        print('\n')
        print('type(frame): ')
        print(type(frame))
        print('\n')

        range_images, camera_projections, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(frame)

        points_return_1, cam_proj_points_ret_1 = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=0)
        points_return_2, cam_proj_points_ret_2 = frame_utils.convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=1)

        # convert_range_image_to_point_cloud returns points as a list of 5 numpy arrays,
        # concatenate the 5 into a single numpy array
        points_return_1 = np.concatenate(points_return_1, axis=0)
        points_return_2 = np.concatenate(points_return_2, axis=0)

        # combine the points from return 1 and return 2 into a single numpt array of all points
        lidarPoints = np.concatenate((points_return_1, points_return_2), axis=0)

        # lidar points is n rows x 3 cols, to use with Plotly, need to change to 3 rows x n cols
        lidarPoints = lidarPoints.transpose()

        ### 3D visualization ######################################################

        s3dPoints = PlotlyGraphObjects.Scatter3d(x=lidarPoints[0], y=lidarPoints[1], z=lidarPoints[2], mode='markers', marker={'size': 1})

        # 3 separate lists for the x, y, and z components of each line
        predXLines = []
        predYLines = []
        predZLines = []
        for lidarLabel in frame.laser_labels:

            cornerPts = getBBoxCornerPoints(lidarLabel.box.center_x, lidarLabel.box.center_y, lidarLabel.box.center_z,
                                            lidarLabel.box.width, lidarLabel.box.length, lidarLabel.box.height,
                                            lidarLabel.box.heading)

            # 4 lines for front surface of box
            addLineToPlotlyLines(cornerPts[0], cornerPts[1], predXLines, predYLines, predZLines)
            addLineToPlotlyLines(cornerPts[1], cornerPts[2], predXLines, predYLines, predZLines)
            addLineToPlotlyLines(cornerPts[2], cornerPts[3], predXLines, predYLines, predZLines)
            addLineToPlotlyLines(cornerPts[3], cornerPts[0], predXLines, predYLines, predZLines)

            # 4 lines between front points and rear points
            addLineToPlotlyLines(cornerPts[0], cornerPts[4], predXLines, predYLines, predZLines)
            addLineToPlotlyLines(cornerPts[1], cornerPts[5], predXLines, predYLines, predZLines)
            addLineToPlotlyLines(cornerPts[2], cornerPts[6], predXLines, predYLines, predZLines)
            addLineToPlotlyLines(cornerPts[3], cornerPts[7], predXLines, predYLines, predZLines)

            # 4 lines for rear surface of box
            addLineToPlotlyLines(cornerPts[4], cornerPts[7], predXLines, predYLines, predZLines)
            addLineToPlotlyLines(cornerPts[5], cornerPts[4], predXLines, predYLines, predZLines)
            addLineToPlotlyLines(cornerPts[6], cornerPts[5], predXLines, predYLines, predZLines)
            addLineToPlotlyLines(cornerPts[7], cornerPts[6], predXLines, predYLines, predZLines)

        # end for

        s3dPredBoxLines = PlotlyGraphObjects.Scatter3d(x=predXLines, y=predYLines, z=predZLines, mode='lines')

        # make and show a plotly Figure object
        plotlyFig = PlotlyGraphObjects.Figure(data=[s3dPoints, s3dPredBoxLines])
        plotlyFig.update_layout(scene_aspectmode='data')

        if not SHOW_PLOTLY_MOUSEOVERS:
            plotlyFig.update_layout(hovermode=False)
            plotlyFig.update_layout(scene = dict(xaxis_showspikes=False,
                                                 yaxis_showspikes=False,
                                                 zaxis_showspikes=False))
        # end if

        plotlyFig.show()

        # only show one frame
        break
    # end for
# end function

def getBBoxCornerPoints(centerX: float, centerY: float, centerZ: float, width: float, length: float, height: float, yaw: float) -> np.ndarray:
    """
    This function is based on:
    https://github.com/argoai/argoverse-api/blob/master/argoverse/data_loading/object_label_record.py#L105

    Calculate the 8 bounding box corners (returned as points inside the egovehicle's frame).
    Returns:
        Numpy array of shape (8,3)
    Corner numbering::
         5------4
         |\\    |\\
         | \\   | \\
         6--\\--7  \\
         \\  \\  \\ \\
     l    \\  1-------0    h
      e    \\ ||   \\ ||   e
       n    \\||    \\||   i
        g    \\2------3    g
         t      width.     h
          h.               t.
    First four corners are the ones facing forward.
    The last four are the ones facing backwards.
    """

    # ToDo: this function works now but has many transposes, try to neaten this up

    xCornerPts = length / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    yCornerPts = width / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    zCornerPts = height / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])

    cornerPts = np.vstack((xCornerPts, yCornerPts, zCornerPts))

    # cornerPts is now 3 rows x 8 cols

    sinYaw = math.sin(yaw)
    cosYaw = math.cos(yaw)

    yawRotMat = np.array([[cosYaw, -1.0 * sinYaw, 0.0],
                          [sinYaw, cosYaw, 0.0],
                          [0.0, 0.0, 1.0]], np.float32)

    # change cornerPts to 8 rows by 3 cols
    cornerPts = cornerPts.transpose()

    # dot product with yaw rotation matrix
    cornerPts = np.dot(cornerPts, yawRotMat)

    # perform translation
    # increment all x values by center x
    cornerPts[:, 0] += centerX
    # increment all y values by center y
    cornerPts[:, 1] += centerY
    # increment all z values by center z
    cornerPts[:, 2] += centerZ

    return cornerPts
# end function

def addLineToPlotlyLines(point1, point2, xLines: List, yLines: List, zLines: List) -> None:
    xLines.append(point1[0])
    xLines.append(point2[0])
    xLines.append(None)

    yLines.append(point1[1])
    yLines.append(point2[1])
    yLines.append(None)

    zLines.append(point1[2])
    zLines.append(point2[2])
    zLines.append(None)
# end function

if __name__ == '__main__':
    main()




