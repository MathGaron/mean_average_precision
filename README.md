# Detection mAP

A simple utility tool to evaluate Bounding box classification task following Pascal VOC [paper](http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf).
To learn about this metric I recommend this excellent blog post by Sancho McCann before reading the paper : [link](https://sanchom.wordpress.com/2011/09/01/precision-recall)

### features
- Simple : numpy and matplotlib are the only dependencies
- Compute a running evaluation : input prediction/ground truth at each frames, no need to save in files
- Plot per class pr-curves with average precision (matplotlib)

![example](https://github.com/MathGaron/mean_average_precision/raw/master/image/pr-curve.png "pr-curves")
An usage example can be found [Here](https://github.com/MathGaron/mean_average_precision/blob/master/mean_average_precision/example.py).


### TODO
- Interpolated average precision

### Contribution
And of course any bugfixes/contribution are welcomed!