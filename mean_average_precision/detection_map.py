import numpy as np
from mean_average_precision.ap_accumulator import APAccumulator
from mean_average_precision.bbox_utils import jaccard
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
        for accumulators, r in zip(self.total_accumulators, self.pr_scale):
            if DEBUG:
                print("Evaluate pr_scale {}".format(r))
            self.evaluate_(accumulators, pred_bb, pred_classes, pred_conf, gt_bb, gt_classes, r, self.overlap_threshold)

    @staticmethod
    def evaluate_(accumulators, pred_bb, pred_classes, pred_conf, gt_bb, gt_classes, confidence_threshold, overlap_threshold=0.5):
        pred_classes = pred_classes.astype(np.int)
        gt_classes = gt_classes.astype(np.int)
        pred_size = pred_classes.shape[0]
        IoU = None
        if pred_size != 0:
            IoU = DetectionMAP.compute_IoU(pred_bb, gt_bb, pred_conf, confidence_threshold)
            # mask irrelevant overlaps
            IoU[IoU < overlap_threshold] = 0

        # Score Gt with no prediction
        for i, acc in enumerate(accumulators):
            qty = DetectionMAP.compute_false_negatives(pred_classes, gt_classes, IoU, i)
            acc.inc_not_predicted(qty)

        # If no prediction are made, no need to continue further
        if len(pred_bb) == 0:
            return

        # Final match : 1 prediction per GT
        for i, acc in enumerate(accumulators):
            qty = DetectionMAP.compute_true_positive(pred_classes, gt_classes, IoU, i)
            acc.inc_good_prediction(qty)
            qty = DetectionMAP.compute_false_positive(pred_classes, pred_conf, confidence_threshold, gt_classes, IoU, i)
            acc.inc_bad_prediction(qty)
        if DEBUG:
            print(accumulators[3])

    @staticmethod
    def compute_IoU(prediction, gt, confidence, confidence_threshold):
        IoU = jaccard(prediction, gt)
        IoU[confidence < confidence_threshold, :] = 0
        return IoU

    @staticmethod
    def compute_false_negatives(pred_cls, gt_cls, IoU, class_index):
        if len(pred_cls) == 0:
            return np.sum(gt_cls == class_index)
        IoU_mask = IoU != 0
        # check only the predictions from class index
        prediction_masks = pred_cls != class_index
        IoU_mask[prediction_masks, :] = False
        # keep only gt of class index
        mask = IoU_mask[:, gt_cls == class_index]
        # sum all gt with no prediction of its class
        return np.sum(np.logical_not(mask.any(axis=0)))

    @staticmethod
    def compute_true_positive(pred_cls, gt_cls, IoU, class_index):
        IoU_mask = IoU != 0
        # check only the predictions from class index
        prediction_masks = pred_cls != class_index
        IoU_mask[prediction_masks, :] = False
        # keep only gt of class index
        mask = IoU_mask[:, gt_cls == class_index]
        # sum all gt with prediction of this class
        return np.sum(mask.any(axis=0))

    @staticmethod
    def compute_false_positive(pred_cls, pred_conf, conf_threshold, gt_cls, IoU, class_index):
        # check if a prediction of other class on class_index gt
        IoU_mask = IoU != 0
        prediction_masks = pred_cls == class_index
        IoU_mask[prediction_masks, :] = False
        mask = IoU_mask[:, gt_cls == class_index]
        FP_predicted_by_other = np.sum(mask.any(axis=0))

        IoU_mask = IoU != 0
        prediction_masks = pred_cls != class_index
        IoU_mask[prediction_masks, :] = False
        gt_masks = gt_cls != class_index
        IoU_mask[:, gt_masks] = False
        # check if more than one prediction on class_index gt
        mask_double = IoU_mask[pred_cls == class_index, :]
        detection_per_gt = np.sum(mask_double, axis=0)
        FP_double = np.sum(detection_per_gt[detection_per_gt > 1] - 1)
        # check if class_index prediction outside of class_index gt
        # total prediction of class_index - prediction matched with class index gt
        detection_per_prediction = np.logical_and(pred_conf >= conf_threshold, pred_cls == class_index)
        FP_predict_other = np.sum(detection_per_prediction) - np.sum(detection_per_gt)
        return FP_double + FP_predict_other + FP_predicted_by_other

    @staticmethod
    def multiple_prediction_on_gt(IoU_mask, gt_classes, accumulators):
        """
        Gt with more than one overlap get False detections
        :param prediction_confidences:
        :param IoU_mask: Mask of valid intersection over union  (np.array)      IoU Shape [n_pred, n_gt]
        :param gt_classes:
        :param accumulators:
        :return: updated version of the IoU mask
        """
        # compute how many prediction per gt
        pred_max = np.sum(IoU_mask, axis=0)
        for i, gt_sum in enumerate(pred_max):
            gt_cls = gt_classes[i]
            if gt_sum > 1:
                for j in range(gt_sum - 1):
                    accumulators[gt_cls].inc_bad_prediction()

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
        grid = int(math.ceil(math.sqrt(self.n_class)))
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