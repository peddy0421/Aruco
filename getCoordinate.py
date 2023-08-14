from calendar import c
from turtle import distance, position
from types import CoroutineType
import numpy as np
import cv2
from cv2 import aruco
import pickle
import math

# ArUco variables
ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_6X6_250)
ARUCO_LENGTH = 0.033

# camera parameters
f = open('./cfg/calibrationC310.pckl', 'rb')
cameraMatrix, distCoeffs, rvecs, tvecs = pickle.load(f)

# Create the arrays and variables we'll use to store info like corners and IDs from images processed
corners_all = []  # Corners discovered in all images processed
ids_all = []  # Aruco ids corresponding to corners discovered
image_size = None  # Determined at runtime
coord = None
rotM = np.zeros(shape=(3, 3))
ang = []

cam = cv2.VideoCapture(0)
img_counter = 0

font = cv2.FONT_HERSHEY_SIMPLEX


def rot2eul(R):
    beta = -np.arcsin(R[2, 0])
    alpha = np.arctan2(R[2, 1]/np.cos(beta), R[2, 2]/np.cos(beta))
    gamma = np.arctan2(R[1, 0]/np.cos(beta), R[0, 0]/np.cos(beta))
    return np.array((alpha, beta, gamma))


def eul2rot(theta):

    R = np.array([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                  [np.sin(theta[2])*np.cos(theta[1]),       np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])
                   * np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                  [-np.sin(theta[1]),                        np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1])]])

    return R


def rotate_marker_corners(rvec, markersize, tvec=None):

    mhalf = markersize / 2.0
    # convert rot vector to rot matrix both do: markerworld -> cam-world
    mrv, jacobian = cv2.Rodrigues(rvec)

    # in markerworld the corners are all in the xy-plane so z is zero at first
    X = mhalf * mrv[:, 0]  # rotate the x = mhalf
    Y = mhalf * mrv[:, 1]  # rotate the y = mhalf
    minusX = X * (-1)
    minusY = Y * (-1)

    # calculate 4 corners of the marker in camworld. corners are enumerated clockwise
    markercorners = []
    markercorners.append(np.add(minusX, Y))  # was upper left in markerworld
    markercorners.append(np.add(X, Y))  # was upper right in markerworld
    markercorners.append(np.add(X, minusY))  # was lower right in markerworld
    # was lower left in markerworld
    markercorners.append(np.add(minusX, minusY))
    # if tvec given, move all by tvec
    if tvec is not None:
        C = tvec  # center of marker in camworld
        for i, mc in enumerate(markercorners):
            markercorners[i] = np.add(C, mc)  # add tvec to each corner
        # print('Vec X, Y, C, dot(X,Y)', X, Y, C, np.dot(X, Y))  # just for debug
    # type needed when used as input to cv2
    markercorners = np.array(markercorners, dtype=np.float32)
    return markercorners, mrv


def getCoord(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # If our image size is unknown, set it now
    # if not image_size:
    #     image_size = gray.shape[::-1]

    # Find aruco markers in the query image
    corners, ids, _ = aruco.detectMarkers(
        image=gray,
        dictionary=ARUCO_DICT)

    # Outline the aruco markers found in our query image
    frame = aruco.drawDetectedMarkers(
        image=frame,
        corners=corners)

    coord = None
    if ids is not None:
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
            corners=corners,
            markerLength=ARUCO_LENGTH,
            cameraMatrix=cameraMatrix,
            distCoeffs=distCoeffs)

        (rvec-tvec).any()

        # print('rvec: {}, tvec: {}'.format(rvec, tvec))
        for i in range(rvec.shape[0]):
            vector = rotate_marker_corners(
                rvec[i, :, :], ARUCO_LENGTH, tvec[i, :, :])
            # vector = cv2.Rodrigues(rvec[i, :, :])
            # vector = np.transpose(vector)
            # position = (-vector*tvec[i, :, :])
            # print('position: ', position)
            # print('vector: ', vector)
            #aruco.drawAxis VS cv2.drawFrameAxes
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs,
                           rvec[i, :, :], tvec[i, :, :], ARUCO_LENGTH/2)

            temp = []

            rotM, jacobian = cv2.Rodrigues(rvec[i])
            ang = rot2eul(rotM)  # yaw, pitch, row

            for j, e in enumerate(ang):
                ang[j] = e/math.pi*180

            if ang[0]<0:
                ang[0]=ang[0]+360
            ang[0] = 180-ang[0]

            temp.append(tvec[i, 0, 0]*1000)  # x    (mm)
            temp.append(tvec[i, 0, 1]*1000)  # y    (mm)
            temp.append(tvec[i, 0, 2]*1000)  # z    (mm)
            temp.append(ang[0])  # rx
            temp.append(ang[1])  # ry
            temp.append(ang[2])  # rz

            coord = []
            coord.append(temp)
        return coord
    else:
        return None


while True:
    ret, frame = cam.read()

    coord = getCoord(frame=frame)

    # cv2.putText(frame, 'Id: ' + str(ids), (5, 64),
    #             font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    if coord is not None:
        print(coord)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)

    if key % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        cam.release()
        cv2.destroyAllWindows()
        break

    elif key % 256 == 32:
        # SPACE pressed
        img_name = "./result/capture{}.jpg".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
