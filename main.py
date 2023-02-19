import cv2 as cv
import numpy as np
import glob

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __truediv__(self, other):
        return Point(self.x / other, self.y / other)

    def __mul__(self, other):
        return Point(self.x * other, self.y * other)


# Global Variables
SQUARE_SIZE: float = 24.23
WIDTH: int = 6
HEIGHT: int = 9
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
chessBoardFlags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE

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
    return y1 + (i - x1)((y2 - y1) / (x2 - x1))


def runCalibration():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((WIDTH * HEIGHT, 3), np.float32)
    objp[:, :2] = np.mgrid[0:WIDTH, 0:HEIGHT].T.reshape(-1, 2)

    for index, obj in enumerate(objp):
        objp[index] = obj * SQUARE_SIZE

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob('images/*.jpg')

    for (index, fname) in enumerate(images):
        # Get Image
        img = cv.imread(fname)

        # Turn image to grayscale
        grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Show image
        cv.imshow('img', img)

        # Try to find corners
        success, corners = cv.findChessboardCorners(grayImg, (WIDTH, HEIGHT), None)

        # If successfully found
        if success and index != 0:
            # Increase corner accuracy and append
            corners = cv.cornerSubPix(grayImg, corners, (11, 11), (-1, -1), criteria)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (WIDTH, HEIGHT), corners, success)

            objpoints.append(objp)
            imgpoints.append(corners)

            cv.imshow('img', img)
            cv.waitKey(5000)
        # If not successfully found
        else:
            print('Rejected')

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
            boardPoints = []
            tR = Point(corners3[0][0], corners3[0][1])
            tL = Point(corners3[1][0], corners3[1][1])
            bR = Point(corners3[2][0], corners3[2][1])
            bL = Point(corners3[3][0], corners3[3][1])

            tRtobR = tR - bR
            tLtobL = tL - bL
            tRtobRd = tRtobR / (WIDTH - 1)
            tLtobLd = tLtobL / (WIDTH - 1)

            for i in range(HEIGHT):
                for j in range(WIDTH):
                    start = tR - tRtobRd * j
                    end = tL - tLtobLd * j
                    step = (end - start) / (HEIGHT - 1)
                    res = start + step * i
                    boardPoints.append((res.x, res.y))

            objpoints.append(objp)
            boardPoints = list(map(lambda x: (np.float32(x[0]), np.float32(x[1])), boardPoints))
            corners4 = np.array(boardPoints).reshape(54, 1, 2)
            # print(corners4)
            # corners5 = cv.cornerSubPix(grayImg, corners4, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners4)

            # Draw and display the corners
            # for i in range(1,54):
            # cv.line(img, tuple(map(int, asdf[i-1])), tuple(map(int, asdf[i])), ((i%5) * 50, 255-(i%5) * 50, 0), 3)

            print(corners4)
            cv.drawChessboardCorners(img, (WIDTH, HEIGHT), corners4, True)
            cv.imshow('img', img)
            cv.waitKey(5000)
            corners3.clear()

    global ret, matrix, distortion
    ret, matrix, distortion, _, _ = cv.calibrateCamera(objpoints, imgpoints, grayImg.shape[::-1], None, None)


def live():
    cap = cv.Videocapture(0, cv.CAP_ANY)

    if not cap.isOpened():
        print("Cannot open camera")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Frame not available")
            break

        grayImg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        success, corners = cv.findChessboardCorners(grayImg, (WIDTH, HEIGHT), None)

        if(success):
            draw(frame)
def draw(frame):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((WIDTH * HEIGHT, 3), np.float32)
    objp[:, :2] = np.mgrid[0:WIDTH, 0:HEIGHT].T.reshape(-1, 2)

    for index, obj in enumerate(objp):
        objp[index] = obj * SQUARE_SIZE

    # Turn image to grayscale
    grayImg = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Show image
    cv.imshow('img', frame)

    # Try to find corners
    success, corners = cv.findChessboardCorners(grayImg, (WIDTH, HEIGHT), chessBoardFlags)

    if (success):
        corners2 = cv.cornerSubPix(grayImg, corners, (11, 11), (-1, -1), criteria)

        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, matrix, distortion)

        finalImg = drawAxis(frame, rvecs, tvecs)
        frame = drawCube(frame, rvecs, tvecs)

        cv.imshow('img', finalImg)
        cv.waitKey(0)


def drawAxis(frame, rvecs, tvecs):
    axis = np.float32([[3 * SQUARE_SIZE, 0, 0], [0, 3 * SQUARE_SIZE, 0], [0, 0, -3 * SQUARE_SIZE], [0, 0, 0]]).reshape(
        -1, 3)

    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, matrix, np.zeros(5))

    origin = tuple(map(int, (imgpts[3].ravel())))

    cv.line(frame, origin, tuple(map(int, (imgpts[0].ravel()))), (255, 0, 0), 3)
    cv.line(frame, origin, tuple(map(int, (imgpts[1].ravel()))), (0, 255, 0), 3)
    cv.line(frame, origin, tuple(map(int, (imgpts[2].ravel()))), (0, 0, 255), 3)

    return frame


def drawCube(frame, rvecs, tvecs):
    vec = 2 * SQUARE_SIZE
    axis = np.float32(
        [[0, vec, 0], [vec, vec, 0], [vec, 0, 0], [0, 0, -vec], [0, vec, -vec], [vec, vec, -vec], [vec, 0, -vec],
         [0, 0, 0]]).reshape(
        -1, 3)

    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, matrix, distortion)

    origin = tuple(map(int, (imgpts[7].ravel())))

    color = (255, 0, 255)

    cv.line(frame, origin, tuple(map(int, (imgpts[0].ravel()))), color, 3)
    cv.line(frame, origin, tuple(map(int, (imgpts[2].ravel()))), color, 3)
    cv.line(frame, origin, tuple(map(int, (imgpts[3].ravel()))), color, 3)

    cv.line(frame, tuple(map(int, (imgpts[1].ravel()))), tuple(map(int, (imgpts[0].ravel()))), color, 3)
    cv.line(frame, tuple(map(int, (imgpts[1].ravel()))), tuple(map(int, (imgpts[2].ravel()))), color, 3)

    cv.line(frame, tuple(map(int, (imgpts[2].ravel()))), tuple(map(int, (imgpts[6].ravel()))), color, 3)

    cv.line(frame, tuple(map(int, (imgpts[3].ravel()))), tuple(map(int, (imgpts[4].ravel()))), color, 3)
    cv.line(frame, tuple(map(int, (imgpts[3].ravel()))), tuple(map(int, (imgpts[6].ravel()))), color, 3)

    cv.line(frame, tuple(map(int, (imgpts[4].ravel()))), tuple(map(int, (imgpts[0].ravel()))), color, 3)

    cv.line(frame, tuple(map(int, (imgpts[5].ravel()))), tuple(map(int, (imgpts[6].ravel()))), color, 3)
    cv.line(frame, tuple(map(int, (imgpts[5].ravel()))), tuple(map(int, (imgpts[4].ravel()))), color, 3)
    cv.line(frame, tuple(map(int, (imgpts[5].ravel()))), tuple(map(int, (imgpts[1].ravel()))), color, 3)


runCalibration()

frame = cv.imread("images/WIN_20220216_15_16_03_Pro.jpg");
draw(frame)

cv.destroyAllWindows()


