# mtcnn-pytorch (use opencv)
pytorch implementation of  face detection algorithm  MTCNN

### Usage MTCNN

Just download the repository and then do this

```
image = cv2.imread('images/test7.jpg')
bounding_boxes, landmarks = detect_faces(image, gpu_id=None)
image = show_bboxes(image, bounding_boxes, landmarks)
cv2.imshow('image', image)
cv2.waitKey()
```

### Requirements

- pytorch >= 0.4
- numpy
- opencv-python

### Credit

This implementation is heavily inspired by:

- https://github.com/polarisZhao/mtcnn-pytorch

### Reference

**MTCNN:** [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878).

