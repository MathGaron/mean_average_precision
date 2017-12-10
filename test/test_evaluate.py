import unittest
import numpy as np

from mean_average_precision.ap_accumulator import APAccumulator
from mean_average_precision.detection_map import DetectionMAP
from mean_average_precision.utils.show_frame import show_frame


class TestNumpyFunctions(unittest.TestCase):
    def setUp(self):
        self.pred = np.array([[0, 0.55, 0.25, 1],
                              [0, 0.65, 0.25, 1.0],
                              [0.5, 0.8, 0.6, 1.0],
                              [0.75, 0, 1.0, 0.45],
                              [0.75, 0.85, 1.0, 1.0],
                              [0, 0, 0.09, 0.09]])
        self.cls = np.array([0, 0, 0, 0, 1, 2])
        self.conf = np.array([1, 0.7, 0.5, 1, 0.75, 1])
        self.gt = np.array([[0, 0.6, 0.2, 1.0],
                            [0.8, 0.9, 1.0, 1.0],
                            [0.7, 0, 1, 0.5],
                            [0, 0, 0.1, 0.1]])
        self.gt_cls = np.array([0, 0, 1, 2])

        #show_frame(self.pred, self.cls, self.conf, self.gt, self.gt_cls)

    def tearDown(self):
        pass

    def test_is_iou_thresholded(self):
        IoU = DetectionMAP.compute_IoU(self.pred, self.gt, self.conf, 0.8)
        valid_IoU = np.argwhere(IoU > 0)
        np.testing.assert_equal(valid_IoU, np.array([[0, 0], [3, 2], [5, 3]]))

    def test_is_FN_incremented_properly(self):
        IoU = DetectionMAP.compute_IoU(self.pred, self.gt, self.conf, 0)
        qty = DetectionMAP.compute_false_negatives(self.cls, self.gt_cls, IoU, 0)
        self.assertEqual(qty, 1)
        qty = DetectionMAP.compute_false_negatives(self.cls, self.gt_cls, IoU, 1)
        self.assertEqual(qty, 1)
        qty = DetectionMAP.compute_false_negatives(self.cls, self.gt_cls, IoU, 2)
        self.assertEqual(qty, 0)

    def test_is_FN_incremented_properly_if_no_prediction(self):
        IoU = None
        qty = DetectionMAP.compute_false_negatives(np.array([]), self.gt_cls, IoU, 0)
        self.assertEqual(qty, 2)
        qty = DetectionMAP.compute_false_negatives(np.array([]), self.gt_cls, IoU, 1)
        self.assertEqual(qty, 1)
        qty = DetectionMAP.compute_false_negatives(np.array([]), self.gt_cls, IoU, 2)
        self.assertEqual(qty, 1)

    def test_FP_is_incremented_with_multiple_gt_detection(self):
        accumulators = [APAccumulator(), APAccumulator()]
        gt_classes = np.array([0, 1])
        IoU = np.array([[0, 0.6], [0.5, 0.6], [1, 0.6]])
        IoU_mask = IoU >= 0.5
        DetectionMAP.multiple_prediction_on_gt(IoU_mask, gt_classes, accumulators)
        self.assertEqual(accumulators[0].FP, 1)
        self.assertEqual(accumulators[1].FP, 2)

    def test_is_TP_incremented_if_prediction(self):
        accumulators = [APAccumulator(), APAccumulator()]
        gt_classes = np.array([0, 1])
        pred_classes = np.array([0, 0, 0])
        IoU = np.array([[0, 0.6], [0.5, 0.6], [1, 0.6]])
        IoU_mask = IoU >= 0.5

        DetectionMAP.good_gt_prediction(IoU_mask, pred_classes, gt_classes, accumulators)
        for i in accumulators:
            print(i)

