import unittest
import numpy as np

from mean_average_precision.detection_map import DetectionMAP
from mean_average_precision.utils.bbox import jaccard
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
        IoU = DetectionMAP.compute_IoU_mask(self.pred, self.gt, 0.7)
        valid_IoU = np.argwhere(IoU)
        np.testing.assert_equal(valid_IoU, np.array([[0, 0], [1, 0], [3, 2], [5, 3]]))

    def test_is_FN_incremented_properly(self):
        mAP = DetectionMAP(3)
        mAP.evaluate(self.pred, self.cls, self.conf, self.gt, self.gt_cls)

        self.assertEqual(mAP.total_accumulators[0][0].FN, 1)
        self.assertEqual(mAP.total_accumulators[0][1].FN, 1)
        self.assertEqual(mAP.total_accumulators[0][2].FN, 0)

    def test_is_FN_incremented_properly_if_no_prediction(self):
        mAP = DetectionMAP(3)
        mAP.evaluate(np.array([]), np.array([]), np.array([]), self.gt, self.gt_cls)
        self.assertEqual(mAP.total_accumulators[0][0].FN, 2)
        self.assertEqual(mAP.total_accumulators[0][1].FN, 1)
        self.assertEqual(mAP.total_accumulators[0][2].FN, 1)

    def test_is_TP_incremented_properly(self):
        mAP = DetectionMAP(3)
        mAP.evaluate(self.pred, self.cls, self.conf, self.gt, self.gt_cls)

        self.assertEqual(mAP.total_accumulators[0][0].TP, 1)
        self.assertEqual(mAP.total_accumulators[0][1].TP, 0)
        self.assertEqual(mAP.total_accumulators[0][2].TP, 1)

    def test_is_FP_incremented_properly_when_away_from_gt(self):
        mAP = DetectionMAP(3)
        mAP.evaluate(self.pred, self.cls, self.conf, self.gt, self.gt_cls)

        self.assertEqual(mAP.total_accumulators[0][0].FP, 3)
        self.assertEqual(mAP.total_accumulators[0][1].FP, 1)
        self.assertEqual(mAP.total_accumulators[0][2].FP, 0)

