import numpy as np
import collections
import dlib
import cv2
import ctypes
import sdl2
from random import randint

# libname = pathlib.Path().absolute() / "libcmult.so"
libname = 'C:/Python312/Lib/site-packages/sdl2dll/dll/SDL2.dll'
c_lib = ctypes.CDLL(libname)

func = getattr(c_lib, "SDL_WarpMouseInWindow", None)
func.argtypes = [ctypes.POINTER(sdl2.SDL_Window), ctypes.c_int, ctypes.c_int]
func.restype = None


# Define a dictionary that maps the indexes of the facial landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = collections.OrderedDict([
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
])


def shape_to_np(shape, dtype="int"):
    # Initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # Loop over the 68 facial landmarks and convert them to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


# Initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Open camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720


def scale_coordinates(x, y, small_window_x, small_window_y, point_x, point_y):
    scaled_x = int((x / small_window_x) * point_x)
    scaled_y = int((y / small_window_y) * point_y)
    return scaled_x, scaled_y


window = sdl2.SDL_CreateWindow(b"Priklad Event Loop", sdl2.SDL_WINDOWPOS_CENTERED, sdl2.SDL_WINDOWPOS_CENTERED, 1920,
                               1080, sdl2.SDL_WINDOW_SHOWN)
event = sdl2.SDL_Event()
while True:
    while sdl2.SDL_PollEvent(ctypes.byref(event)) != 0:
        if event.type == sdl2.SDL_KEYDOWN:  # event: keydown
            if event.key.keysym.sym == sdl2.SDLK_k:
                x, y = randint(0, 1920), randint(0, 1080)
                func(window, x, y)
        if event.type == sdl2.SDL_QUIT:
            running = False
            break
    sdl2.SDL_Delay(10)
    # Read frame from camera
    (grabbed, image) = camera.read()
    if not grabbed:
        break
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("gray.jpg", gray)

    # Detect faces
    rects = detector(gray, 1)

    # Loop over the face detections
    for rect in rects:
        # Find facial landmarks
        shape = predictor(gray, rect)
        shapenp = shape_to_np(shape)

        # Loop over the facial landmarks for eyes
        result = []
        for name, (i, j) in FACIAL_LANDMARKS_IDXS.items():
            pts = shapenp[i:j]
            hull = cv2.convexHull(pts)
            # Extract ROI corresponding to the convex hull
            x, y, w, h = cv2.boundingRect(hull)
            # sprav masku oka
            mask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.drawContours(mask, [hull], -1, (255, 255, 255), -1)
            # zmensi masku
            kernel = np.ones((1, 1), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            # aplikuj masku na obrazok (vyrez z obrazku oko)
            output = cv2.bitwise_and(image, image, mask=mask)
            # okolie oka vyfarbi bielou
            output[mask == 0] = (255, 255, 255)
            roi = output[y:y + h, x:x + w]
            cv2.imwrite("oko.jpg", roi)
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("Sedeoko.jpg", roi_gray)
            # Apply Gaussian blur to reduce noise
            roi_blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)
            _, thresh_otsu = cv2.threshold(roi_blurred, 60, 255, cv2.THRESH_BINARY_INV)
            # Write the thresholded image
            cv2.imwrite("oko-thresh-otsu.jpg", thresh_otsu)
            contours, _ = cv2.findContours(thresh_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                cnt = sorted(contours, key = cv2.contourArea, reverse=True)

                M = cv2.moments(cnt[0])
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    contour = cv2.drawContours(image, [cnt[0] + (x, y)], -1, (0, 255, 0), 1)  # Draw pupil contour
                    cv2.circle(image, (cX + x, cY + y), 3, (0, 0, 255), -1)  # Draw red dot at centroid position
                    cv2.imwrite("oko-kontura-otsu.jpg", contour)
                    pX = cX / w
                    pY = cY / h
                    result.append((pX, pY))

        if len(result) > 0:
            finalX, finalY = result[0][0], result[0][1]
            if len(result) == 2:
                finalX = (result[0][0] + result[1][0]) / 2
                finalY = (result[0][1] + result[1][1]) / 2
            func(window, int(finalX * 1920), int(finalY * 1080))

    cv2.imwrite("contour.jpg", image)

    cv2.imshow("Image", image)
    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

sdl2.SDL_DestroyWindow(window)
sdl2.SDL_Quit()
# Release resources
camera.release()
cv2.destroyAllWindows()