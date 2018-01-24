import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def show_frame(pred_bb, pred_classes, pred_conf, gt_bb, gt_classes, background=np.zeros((500, 500, 3)), show_confidence=True):
    """
    Plot the boundingboxes
    :param pred_bb: (np.array)      Predicted Bounding Boxes [x1, y1, x2, y2] :     Shape [n_pred, 4]
    :param pred_classes: (np.array) Predicted Classes :                             Shape [n_pred]
    :param pred_conf: (np.array)    Predicted Confidences [0.-1.] :                 Shape [n_pred]
    :param gt_bb: (np.array)        Ground Truth Bounding Boxes [x1, y1, x2, y2] :  Shape [n_gt, 4]
    :param gt_classes: (np.array)   Ground Truth Classes :                          Shape [n_gt]
    :return:
    """
    n_pred = pred_bb.shape[0]
    n_gt = gt_bb.shape[0]
    n_class = np.max(np.append(pred_classes, gt_classes)) + 1
    h, w, c = background.shape

    ax = plt.subplot("111")
    ax.imshow(background)
    cmap = plt.cm.get_cmap('hsv')

    confidence_alpha = pred_conf.copy()
    if not show_confidence:
        confidence_alpha.fill(1)

    for i in range(n_pred):
        x1 = pred_bb[i, 0] * w
        y1 = pred_bb[i, 1] * h
        x2 = pred_bb[i, 2] * w
        y2 = pred_bb[i, 3] * h
        rect_w = x2 - x1
        rect_h = y2 - y1
        print(x1, y1)
        ax.add_patch(patches.Rectangle((x1, y1), rect_w, rect_h,
                                       fill=False,
                                       edgecolor=cmap(float(pred_classes[i]) / n_class),
                                       linestyle='dashdot',
                                       alpha=confidence_alpha[i]))

    for i in range(n_gt):
        x1 = gt_bb[i, 0] * w
        y1 = gt_bb[i, 1] * h
        x2 = gt_bb[i, 2] * w
        y2 = gt_bb[i, 3] * h
        rect_w = x2 - x1
        rect_h = y2 - y1
        ax.add_patch(patches.Rectangle((x1, y1), rect_w, rect_h,
                                       fill=False,
                                       edgecolor=cmap(float(gt_classes[i]) / n_class)))

    legend_handles = []
    for i in range(n_class):
        legend_handles.append(patches.Patch(color=cmap(float(i) / n_class), label="class : {}".format(i)))
    ax.legend(handles=legend_handles)
    plt.show()

