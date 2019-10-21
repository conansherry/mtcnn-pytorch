from utils.visualization_utils import show_bboxes
from src.detector import MTCNNFaceDetector
import cv2

def main():
    detector = MTCNNFaceDetector(gpu_id=0)

    image = cv2.imread(r'F:\workspace\face-alignment\test\assets\aflw-test.jpg')
    bounding_boxes, landmarks = detector.forward(image)
    image = show_bboxes(image, bounding_boxes, landmarks)
    cv2.imshow('image', image)
    cv2.waitKey()

    # image = cv2.imread('images/test5.jpg')
    # bounding_boxes, landmarks = detector.forward(image)
    # image = show_bboxes(image, bounding_boxes, landmarks)
    # cv2.imshow('image', image)
    # cv2.waitKey()

if __name__ == "__main__":
    main()
