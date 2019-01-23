from src.detector import detect_faces
from utils.visualization_utils import show_bboxes
import cv2

def main():
    image = cv2.imread('images/test7.jpg')
    bounding_boxes, landmarks = detect_faces(image, gpu_id=None)
    image = show_bboxes(image, bounding_boxes, landmarks)
    cv2.imshow('image', image)
    cv2.waitKey()

if __name__ == "__main__":
    main()
