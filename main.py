import cv2 as cv
import numpy as np
import glob

# Global Variables
SQUARE_SIZE: int = 24
WIDTH: int = 9
HEIGHT: int = 6
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

corners3 = []

def click_event(event, x, y, flags, params):
    global corners3
    if event == cv.EVENT_LBUTTONDOWN:
        corners3.append([x, y])

def interpolate(i, x, y):
    x1 = x.x
    x2 = y.x
    y1 = x.y
    y2 = y.y
    return y1 + (i - x1)((y2-y1)/(x2-x1))

def runCalibration():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((WIDTH * HEIGHT, 3), np.float32)
    objp[:, :2] = np.mgrid[0:HEIGHT, 0:WIDTH].T.reshape(-1, 2)

    for index, obj in enumerate(objp):
        objp[index] = obj * SQUARE_SIZE

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob('images/*.jpg')

    for fname in images:
        # Get Image
        img = cv.imread(fname)

        # Turn image to grayscale
        grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Show image
        cv.imshow('img', img)

        # Try to find corners
        success, corners = cv.findChessboardCorners(grayImg, (WIDTH, HEIGHT), None)

        # If successfully found
        if True:
            # Increase corner accuracy and append
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(grayImg, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, (WIDTH, HEIGHT), corners2, success)
            cv.imshow('img', img)
            cv.waitKey(500)
        # If not successfully found
        else:
            cv.imshow('img', img)

            # Set mouse click listening event
            cv.setMouseCallback('img', click_event)

            # Gather 4 corner clicks
            while True:
                cv.imshow('img', img)
                cv.waitKey(1)

                if len(corners3) == 4:
                    break

            # TODO: interpolate

            # Increase corner accuracy and append
            objpoints.append(objp)
            corners4 = np.array(corners3)
            print(corners4)
            corners5 = cv.cornerSubPix(grayImg, corners4, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners5)

            # Draw and display the corners
            cv.drawChessboardCorners(img, (WIDTH, HEIGHT), corners5)
            cv.imshow('img', img)
            cv.waitKey(500)
            corners3.clear()

    global ret, matrix, distortion, r_vecs, t_vecs
    ret, matrix, distortion, r_vecs, t_vecs = cv.calibrateCamera(objpoints, imgpoints, grayImg.shape[:2], None, None)

def draw(frame):
    # Read frame
    img = cv.imread(frame)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((WIDTH * HEIGHT, 3), np.float32)
    objp[:, :2] = np.mgrid[0:HEIGHT, 0:WIDTH].T.reshape(-1, 2)

    for index, obj in enumerate(objp):
        objp[index] = obj * SQUARE_SIZE

    # Turn image to grayscale
    grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Show image
    cv.imshow('img', img)

    # Try to find corners
    success, corners = cv.findChessboardCorners(grayImg, (WIDTH, HEIGHT), None)

    if(success):
        corners2 = cv.cornerSubPix(grayImg, corners, (11, 11), (-1, -1), criteria)

        rvecs, tvecs = cv.solvePnP(objp, corners2, matrix, distortion)

        frame = drawAxis(frame,  rvecs, tvecs)
        #frame = drawCube(frame, rvecs, tvecs)

def drawAxis(frame, rvecs, tvecs):
    axis = np.float32([[3,0,0] * SQUARE_SIZE, [0,3,0] * SQUARE_SIZE, [0,0,-3] * SQUARE_SIZE, [0,0,0]]).reshape(-1,4)

    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, matrix, distortion)

    print(imgpts)


runCalibration()


images = glob.glob('images/*.jpg')
draw(images[0])

cv.destroyAllWindows()