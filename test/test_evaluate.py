import unittest
import numpy as np

from mean_average_precision.ap_accumulator import APAccumulator
from mean_average_precision.bbox_utils import jaccard
from mean_average_precision.detection_map import DetectionMAP


class TestNumpyFunctions(unittest.TestCase):
    def setUp(self):
        self.pred = np.array([[],
                              [],
                              []])
        self.cls = np.array([0, 1, 2])
        self.conf = np.array([0.5, 0.7, 0.1])
        self.gt = np.array([[],
                            []])
        self.gt_cls = np.array([0, 0, 1, 2])

    def tearDown(self):
        pass

    def test_is_iou_thresholded(self):
        prediction = np.array([[0.5, 0.5, 0.6, 0.6]])
        gt = np.array([[0.5, 0.5, 0.6, 0.6]])
        confidence = np.array([0.2])
        IoU = DetectionMAP.compute_IoU(prediction, gt, confidence, 0.3)
        np.testing.assert_equal(IoU, np.array([[0.]]))
        IoU = DetectionMAP.compute_IoU(prediction, gt, confidence, 0.1)
        np.testing.assert_equal(IoU, np.array([[1.]]))

    def test_is_FN_incremented_properly(self):
        pass

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

