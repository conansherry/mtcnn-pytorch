import cv2

def show_bboxes(img, bounding_boxes, facial_landmarks=[]):
    """
        Draw bounding boxes and facial landmarks.
    """
    img_copy = img.copy()

    for box_score in bounding_boxes:
        cv2.rectangle(img_copy, (int(box_score[0]), int(box_score[1])),
                      (int(box_score[2]), int(box_score[3])),
                      (0, 255, 0),
                      2)

    for pt in facial_landmarks:
        for i in range(5):
            cv2.circle(img_copy, (int(pt[i]), int(pt[i + 5])), 1, (255, 0, 0), 2)

    return img_copy
