import numpy as np
from ap_accumulator import APAccumulator
from bbox_utils import jaccard
import math
import matplotlib.pyplot as plt


class DetectionMAP:
    def __init__(self, n_class, pr_samples=11, overlap_threshold=0.5):
        """
        Running computation of average precision of n_class in a bounding box + classification task
        :param n_class:             quantity of class
        :param pr_samples:          quantification of threshold for pr curve
        :param overlap_threshold:   minimum overlap threshold
        """
        self.n_class = n_class
        self.overlap_threshold = overlap_threshold
        self.pr_scale = np.linspace(0, 1, pr_samples)
        self.total_accumulators = []
        self.reset_accumulators()

    def reset_accumulators(self):
        self.total_accumulators = []
        for i in range(len(self.pr_scale)):
            class_accumulators = []
            for j in range(self.n_class):
                class_accumulators.append(APAccumulator())
            self.total_accumulators.append(class_accumulators)

    def evaluate(self, pred_bb, pred_classes, pred_conf, gt_bb, gt_classes):
        """
        Update the accumulator for the running mAP evaluation.
        For exemple, this can be called for each images
        :param pred_bb: (np.array)      Predicted Bounding Boxes [x1, y1, x2, y2] :     Shape [n_pred, 4]
        :param pred_classes: (np.array) Predicted Classes :                             Shape [n_pred]
        :param pred_conf: (np.array)    Predicted Confidences [0.-1.] :                 Shape [n_pred]
        :param gt_bb: (np.array)        Ground Truth Bounding Boxes [x1, y1, x2, y2] :  Shape [n_gt, 4]
        :param gt_classes: (np.array)   Ground Truth Classes :                          Shape [n_gt]
        :return:
        """
        for accumulators, r in zip(self.total_accumulators, self.pr_scale):
            self.evaluate_(accumulators, pred_bb, pred_classes, pred_conf, gt_bb, gt_classes, r, self.overlap_threshold)

    @staticmethod
    def evaluate_(accumulators, pred_bb, pred_classes, pred_conf, gt_bb, gt_classes, threshold, overlap_threshold=0.5):
        pred_classes = pred_classes.astype(np.int)
        gt_classes = gt_classes.astype(np.int)
        gt_size = gt_classes.shape[0]
        IoU = jaccard(pred_bb, gt_bb)

        # Remove IoU with low confidence
        not_confident_mask = pred_conf < threshold
        IoU[not_confident_mask, :] = 0
        # mask irrelevant overlaps
        IoU_mask = IoU >= overlap_threshold

        # Gt with more than one overlap get False detections
        pred_conf_grid = np.repeat(pred_conf[:, np.newaxis], gt_size, axis=1)
        pred_conf_grid[np.bitwise_not(IoU_mask)] = 0
        pred_max = np.max(pred_conf_grid, axis=0)
        invalid = pred_conf_grid == 0
        doubles = pred_conf_grid == pred_max[np.newaxis, :]
        double_mask = np.bitwise_not(np.bitwise_or(invalid, doubles))
        double_match = np.argwhere(double_mask)
        for match in double_match:
            gt_cls = gt_classes[match[1]]
            accumulators[gt_cls].inc_bad_prediction()

        # Final match : 1 prediction per GT
        IoU_mask = np.bitwise_and(np.bitwise_not(invalid), doubles)
        bb_match = np.argwhere(IoU_mask)  # Index [pred, gt]
        for match in bb_match:
            predicted_cls = pred_classes[match[0]]
            gt_cls = gt_classes[match[1]]
            if gt_cls == predicted_cls:
                accumulators[gt_cls].inc_good_prediction()
            else:
                accumulators[gt_cls].inc_bad_prediction()

        # Score Gt with no prediction
        unclassified_mask = np.max(IoU, axis=0) < overlap_threshold
        for cls in gt_classes[unclassified_mask]:
            accumulators[cls].inc_not_predicted()

        # Score prediction too far from GT
        lonely_boundingbox = np.max(IoU, axis=1) < overlap_threshold
        lonely_detection = np.bitwise_and(lonely_boundingbox, np.bitwise_not(not_confident_mask))
        for cls in pred_classes[lonely_detection]:
            accumulators[cls].inc_bad_prediction()

    def compute_ap(self, cls_idx):
        """
        Compute average precision of a particular classes (cls_idx)
        :param cls:
        :return:
        """
        precisions = []
        recalls = []
        for iteration in self.total_accumulators[::-1]:
            precisions.append(iteration[cls_idx].precision)
            recalls.append(iteration[cls_idx].recall)
        previous_recall = 0
        average_precision = 0
        for precision, recall in zip(precisions, recalls):
            average_precision += precision * (recall - previous_recall)
            previous_recall = recall
        return average_precision

    def plot(self):
        """
        Plot all pr-curves for each classes
        :return:
        """
        grid = math.ceil(math.sqrt(self.n_class))
        fig, axes = plt.subplots(nrows=grid, ncols=grid)
        mean_average_precision = []
        # TODO: data structure not optimal for this operation...
        for i, ax in enumerate(axes.flat):
            if i > self.n_class - 1:
                break
            precisions = []
            recalls = []
            for acc in self.total_accumulators:
                precisions.append(acc[i].precision)
                recalls.append(acc[i].recall)
            average_precision = self.compute_ap(i)
            mean_average_precision.append(average_precision)
            ax.step(recalls, precisions, color='b', alpha=0.2,
                     where='post')
            ax.fill_between(recalls, precisions, step='post', alpha=0.2,
                             color='b')
            ax.set_ylim([0.0, 1.05])
            ax.set_xlim([0.0, 1.0])
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('cls {0:} : AUC={1:0.2f}'.format(i, average_precision))
        plt.suptitle("Mean average precision : {:0.2f}".format(sum(mean_average_precision)/len(mean_average_precision)))
        fig.tight_layout()
        plt.show()