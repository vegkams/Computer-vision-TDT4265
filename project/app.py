# coding=utf-8
import cv2
import filters
import time
from managers import WindowManager, CaptureManager


class Cameo(object):
    def __init__(self):
        self._windowManager = WindowManager('App', self.onKeypress)

        # Web cam
        #self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, True)
        # Read from folder
        self._captureManager = CaptureManager(cv2.VideoCapture("FullIJCNN2013/00%03d.ppm"), self._windowManager, False)

        self._sharpenFilter = filters.SharpenFilter()
        self._findEdgesFilter = filters.FindEdgesFilter()
        self._blurFilter = filters.BlurFilter()
        self._embossFilter = filters.EmbossFilter()


        self._edgeKernelSize = 127
        self._blurKernelSize = 13
        self._cannyThresholdValue1 = 100
        self._cannyThresholdValue2 = 200
        self._binaryThresh = 127


    def run(self):
        """Run the main loop."""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            ###### Filter testing #########

            self.filterTester(frame)

            ###### Filter testing #########

            time.sleep(2)

            self._captureManager.exitFrame()


            self._windowManager.processEvents()


    def filterTester(self, frame):

        # self._captureManager._frame = filters.g_hpf(frame, self._edgeKernelSize)

        # filters.strokeEdges(frame, frame, self_edgeKernelSize)

        # self._captureManager._frame = filters.edgeLaplacian(frame, self._edgeKernelSize, self._blurKernelSize)

        # filters.edgeLaplacianColor(frame, self._edgeKernelSize, self._blurKernelSize)

        # self._sharpenFilter.apply(frame, frame)

        # self._findEdgesFilter.apply(frame, frame)

        # self._blurFilter.apply(frame, frame)

        # self._embossFilter.apply(frame, frame)

        # frame = cv2.medianBlur(frame, 11)

        # self._captureManager._frame = filters.cannyEdge(frame, self._cannyThresholdValue1, self._cannyThresholdValue2)

        # self._captureManager._frame = filters.contourDetection(frame)

        # self._captureManager._frame = filters.contourDetectionCircleSquare(frame)

        # self._captureManager._frame = filters.contourDetectionPolygon(frame)

        #filters.circleDetection(frame)

        # self._captureManager._frame = filters.watershed(frame, self._edgeKernelSize, self._blurKernelSize)

        # self._captureManager._frame = filters.autoThresh(frame,self._edgeKernelSize ,self._blurKernelSize)

        # self._captureManager._frame = filters.autoThreshMorph(frame,self._edgeKernelSize ,self._blurKernelSize)


    def onKeypress(self, keycode):
        """Handle a keypress.
            
            space  -> Take a screenshot.
            tab    -> Start/stop recording a screencast.
            escape -> Quit.
            
            """
        if keycode == 32:  # space
            self._captureManager.writeImage('screenshot.png')
        elif keycode == 9:  # tab
            if not self._captureManager.isWritingVideo:
                # self._captureManager.startWritingVideo('screencast.avi')
                self._captureManager.startWritingVideo('screencast.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 27:  # escape
            self._windowManager.destroyWindow()
        elif keycode == 43:  # +
            if self._edgeKernelSize >= 1:
                self._edgeKernelSize = self._edgeKernelSize + 2
                print("Edge kernel size: ", self._edgeKernelSize)
        elif keycode == ord('å'):  # å
            if self._edgeKernelSize > 1:
                self._edgeKernelSize = self._edgeKernelSize - 2
                print("Edge kernel size: ", self._edgeKernelSize)
        elif keycode == 48:  # 0
            if self._blurKernelSize >= 1:
                self._blurKernelSize = self._blurKernelSize + 2
                print("Blur kernel size: ", self._blurKernelSize)
        elif keycode == 112:  # p
            if self._blurKernelSize > 1:
                self._blurKernelSize = self._blurKernelSize - 2
                print("Blur kernel size: ", self._blurKernelSize)
        elif keycode == 57:  # 9
            if self._cannyThresholdValue2 >= 0:
                self._cannyThresholdValue2 = self._cannyThresholdValue2 + 10
                print("Canny treshold 2 : ", self._cannyThresholdValue2)
        elif keycode == 111:  # o
            if self._cannyThresholdValue2 > 1:
                self._cannyThresholdValue2 = self._cannyThresholdValue2 - 10
                print("Canny treshold 2 : ", self._cannyThresholdValue2)
        elif keycode == 56:  # 8
            if self._cannyThresholdValue1 >= 0:
                self._cannyThresholdValue1 = self._cannyThresholdValue1 + 10
                print("Canny treshold 1 : ", self._cannyThresholdValue1)
        elif keycode == 105:  # i
            if self._cannyThresholdValue1 > 1:
                self._cannyThresholdValue1 = self._cannyThresholdValue1 - 10
                print("Canny treshold 1 : ", self._cannyThresholdValue1)
        elif keycode == ord('7'):  # 7
            if self._binaryThresh >= 0:
                self._binaryThresh = self._binaryThresh + 1
                print("Binary treshold : ", self._binaryThresh)
        elif keycode == ord('u'):  # u
            if self._binaryThresh > 1:
                self._binaryThresh = self._binaryThresh - 1
                print("Binary threshold : ", self._binaryThresh)


if __name__ == "__main__":
    Cameo().run()
