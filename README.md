# Detection mAP

A simple utility tool to evaluate Bounding box classification task following Pascal VOC [paper](http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf).

To learn about this metric I recommend this excellent blog post by Sancho McCann before reading the paper : [link](https://sanchom.wordpress.com/2011/09/01/precision-recall)

**Note that the method is not compared with the original VOC implementation! (See Todo)**

## features
- Simple : numpy and matplotlib are the only dependencies
- Compute a running evaluation : input prediction/ground truth at each frames, no need to save in files
- Plot (matplotlib) per class pr-curves with interpolated average precision (default) or average precision

## Method
### Multiclass mAP
Handle every class as one against the others. (x against z)
- True positive (TP):
    - Gt x predicted as x
- False positive (FP):
    - Prediction x if Gt x has already a TP prediction
    - Prediction x not overlapping any Gt x
- False negative (FN):
    - Gt x not predicted as x
### Example frame
![example](https://github.com/MathGaron/mean_average_precision/raw/master/image/example_frame.png "example frame")

## Code
All you need is your predicted bounding boxes with class and confidence score and the ground truth bounding boxes with their classes.

### [main loop](https://github.com/MathGaron/mean_average_precision/blob/master/mean_average_precision/example.py) :
```python
  frames = [(pred_bb1, pred_cls1, pred_conf1, gt_bb1, gt_cls1),
            (pred_bb2, pred_cls2, pred_conf2, gt_bb2, gt_cls2),
            (pred_bb3, pred_cls3, pred_conf3, gt_bb3, gt_cls3)]
  n_class = 7

  mAP = DetectionMAP(n_class)
  for frame in frames:
      mAP.evaluate(*frame)

  mAP.plot()
  plt.show() # or plt.savefig(path)
```
In this example a frame is a tuple containing:
- Predicted bounding boxes :  numpy array [n, 4]
- Predicted classes:          numpy array [n]
- Predicted confidences:      numpy array [n]
- Ground truth bounding boxes:numpy array [m, 4]
- Ground truth classes:       numpy array [m]

Note that the bounding boxes are represented as two corners points : [x1, y1, x2, y2]

![example](https://github.com/MathGaron/mean_average_precision/raw/master/image/pr-curve.png "pr-curves")

### TODO
- ~~Interpolated average precision~~
- Test against [VOC matlab implementation](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html)

### Contribution
And of course any bugfixes/contribution are always welcome!
