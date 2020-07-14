import cv2
import subprocess
from hand_face_detectors import CVLibDetector
from hand_face_detectors import TSDetector
from hand_face_detectors import addObjs
from hand_face_detectors import objects_intersect


class Hand_detection(object):
    def __init__(self):
        pass

    def process(self, frame):
        # return img.transpose(Image.FLIP_LEFT_RIGHT)
        if frame is not None:
            FaceDetector = CVLibDetector()
            HandsDetector = TSDetector()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hands = HandsDetector.detect(rgb)
            face = FaceDetector.detect(rgb)

            # add detected face and hands to frame
            detect_img = addObjs(rgb, hands)
            detect_img = addObjs(detect_img, face, color=(0, 255, 0))

        return detect_img;




# def detect():
#     # access webcam
#     capture = cv2.VideoCapture(1)
#     # set dimensions
#     capture.set(3, 1280 / 2)
#     capture.set(4, 1024 / 2)

#     # set class

#     while (True):
#         # capture frame by frame
#         ret, frame = capture.read()
#         # change frame color
#         if frame is not None:
#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#             hands = HandsDetector.detect(rgb)
#             face = FaceDetector.detect(rgb)

#             # add detected face and hands to frame
#             detect_img = addObjs(rgb, hands)
#             detect_img = addObjs(detect_img, face, color=(0, 255, 0))

#             # display frame with added objects
#             cv2.imshow('frame', cv2.cvtColor(detect_img, cv2.COLOR_RGB2BGR))

#             if objects_intersect(face, hands):
#                 try:
#                     subprocess.call(["afplay", "beepaudio.wav"])
#                 except:
#                     pass

#             # end stream if user types in 'q'
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     # release capture
#     capture.release()
#     cv2.destroyAllWindows()

#     print("bye")


# if __name__ == '__main__':
#     detect()
