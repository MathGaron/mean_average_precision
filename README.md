# Detection mAP

A simple utility tool to evaluate Bounding box classification task following Pascal VOC [paper](http://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf).

To learn about this metric I recommend this blog post by Sancho McCann before reading the paper : [link](https://sanchom.wordpress.com/2011/09/01/precision-recall)

#### features
Let the user compute a running mAP evaluation, which mean that it can be embedded in a loop and fed with prediction and ground truths.

The user can plot a PR-curve for each class with the per class average precision:

![example](https://github.com/mathgaron/mean_average_precision/image/pr-curve.png "pr-curves")


An usage example can be found in mean_average_precision/example.py

#### Contribution
And of course any bugfixes/contribution are welcomed!