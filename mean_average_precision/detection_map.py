import numpy as np
from mean_average_precision.ap_accumulator import APAccumulator
from mean_average_precision.utils.bbox import jaccard
import math
import matplotlib.pyplot as plt

DEBUG = False


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
        """
        Reset the accumulators state
        TODO this is hard to follow... should use a better data structure
        total_accumulators : list of list of accumulators at each pr_scale for each class
        :return:
        """
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

        if pred_bb.ndim == 1:
            pred_bb = np.repeat(pred_bb[:, np.newaxis], 4, axis=1)
        IoUmask = None
        if len(pred_bb) > 0:
            IoUmask = self.compute_IoU_mask(pred_bb, gt_bb, self.overlap_threshold)
        for accumulators, r in zip(self.total_accumulators, self.pr_scale):
            if DEBUG:
                print("Evaluate pr_scale {}".format(r))
            self.evaluate_(IoUmask, accumulators, pred_classes, pred_conf, gt_classes, r)

    @staticmethod
    def evaluate_(IoUmask, accumulators, pred_classes, pred_conf, gt_classes, confidence_threshold):
        pred_classes = pred_classes.astype(np.int)
        gt_classes = gt_classes.astype(np.int)

        for i, acc in enumerate(accumulators):
            gt_number = np.sum(gt_classes == i)
            pred_mask = np.logical_and(pred_classes == i, pred_conf >= confidence_threshold)
            pred_number = np.sum(pred_mask)
            if pred_number == 0:
                acc.inc_not_predicted(gt_number)
                continue

            IoU1 = IoUmask[pred_mask, :]
            mask = IoU1[:, gt_classes == i]

            tp = DetectionMAP.compute_true_positive(mask)
            fp = pred_number - tp
            fn = gt_number - tp
            acc.inc_good_prediction(tp)
            acc.inc_not_predicted(fn)
            acc.inc_bad_prediction(fp)

    @staticmethod
    def compute_IoU_mask(prediction, gt, overlap_threshold):
        IoU = jaccard(prediction, gt)
        # for each prediction select gt with the largest IoU and ignore the others
        for i in range(len(prediction)):
            maxj = IoU[i, :].argmax()
            IoU[i, :maxj] = 0
            IoU[i, (maxj + 1):] = 0
        # make a mask of all "matched" predictions vs gt
        return IoU >= overlap_threshold

    @staticmethod
    def compute_true_positive(mask):
        # sum all gt with prediction of its class
        return np.sum(mask.any(axis=0))

    def compute_ap(self, precisions, recalls):
        """
        Compute average precision of a particular classes (cls_idx)
        :param cls:
        :return:
        """
        previous_recall = 0
        average_precision = 0
        for precision, recall in zip(precisions[::-1], recalls[::-1]):
            average_precision += precision * (recall - previous_recall)
            previous_recall = recall
        return average_precision

    def compute_precision_recall_(self, class_index, interpolated=True):
        precisions = []
        recalls = []
        for acc in self.total_accumulators:
            precisions.append(acc[class_index].precision)
            recalls.append(acc[class_index].recall)

        if interpolated:
            interpolated_precision = []
            for precision in precisions:
                last_max = 0
                if interpolated_precision:
                    last_max = max(interpolated_precision)
                interpolated_precision.append(max(precision, last_max))
            precisions = interpolated_precision
        return precisions, recalls

    def plot_pr(self, ax, class_name, precisions, recalls, average_precision):
        ax.step(recalls, precisions, color='b', alpha=0.2,
                where='post')
        ax.fill_between(recalls, precisions, step='post', alpha=0.2,
                        color='b')
        ax.set_ylim([0.0, 1.05])
        ax.set_xlim([0.0, 1.0])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('{0:} : AUC={1:0.2f}'.format(class_name, average_precision))

    def plot(self, interpolated=True, class_names=None):
        """
        Plot all pr-curves for each classes
        :param interpolated: will compute the interpolated curve
        :return:
        """
        grid = int(math.ceil(math.sqrt(self.n_class)))
        fig, axes = plt.subplots(nrows=grid, ncols=grid)
        mean_average_precision = []
        # TODO: data structure not optimal for this operation...
        for i, ax in enumerate(axes.flat):
            if i > self.n_class - 1:
                break
            precisions, recalls = self.compute_precision_recall_(i, interpolated)
            average_precision = self.compute_ap(precisions, recalls)
            class_name = class_names[i] if class_names else "Class {}".format(i)
            self.plot_pr(ax, class_name, precisions, recalls, average_precision)
            mean_average_precision.append(average_precision)

        plt.suptitle("Mean average precision : {:0.2f}".format(sum(mean_average_precision)/len(mean_average_precision)))
        fig.tight_layout()
