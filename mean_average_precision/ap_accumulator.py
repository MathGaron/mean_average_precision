"""
    Simple accumulator class that keeps track of True positive, False positive and False negative
    to compute precision and recall of a certain class
"""


class APAccumulator:
    def __init__(self):
        self.TP, self.FP, self.FN = 0, 0, 0

    def inc_good_prediction(self, value=1):
        self.TP += value

    def inc_bad_prediction(self, value=1):
        self.FP += value

    def inc_not_predicted(self, value=1):
        self.FN += value

    @property
    def precision(self):
        total_predicted = self.TP + self.FP
        if total_predicted == 0:
            total_gt = self.TP + self.FN
            if total_gt == 0:
                return 1.
            else:
                return 0.
        return float(self.TP) / total_predicted

    @property
    def recall(self):
        total_gt = self.TP + self.FN
        if total_gt == 0:
            return 1.
        return float(self.TP) / total_gt

    def __str__(self):
        str = ""
        str += "True positives : {}\n".format(self.TP)
        str += "False positives : {}\n".format(self.FP)
        str += "False Negatives : {}\n".format(self.FN)
        str += "Precision : {}\n".format(self.precision)
        str += "Recall : {}\n".format(self.recall)
        return str