import numpy as np

from utils.box_utils import jaccard
from inference.infer_utils import extract_pred_matrix

REJECT_SURE_MR_REASON = 'Found real face in picture'
REVIEW_FAKE_REASON = 'Found fake face in image'
REVIEW_MR_REASON = 'Found inconfident face in image'
ACCEPT_REASON = 'Image passed all the test'


class RulesInference():
    """
    Rules to device an image Reject, Review or Accept
    """
    def __init__(self,
                mr_review_threshold,
                mr_reject_threshold,
                no_mr_review_threshold,
                no_mr_reject_threshold,
                fake_threshold,
                human_threshold,
                fake_overlap_real):
        ##TODO: Modify this for text (if we can, make this dynamic)
        self.mr_review_threshold = mr_review_threshold
        self.mr_reject_threshold = mr_reject_threshold
        self.no_mr_review_threshold = no_mr_review_threshold
        self.no_mr_reject_threshold = no_mr_reject_threshold
        self.fake_threshold = fake_threshold
        self.human_threshold = human_threshold
        self.fake_overlap_real = fake_overlap_real

    def is_detection_rejected(self, all_det_preds):
        '''
        Rules to decide image to be rejected
        '''
        pred_matrix = extract_pred_matrix(all_det_preds)
        ##TODO: Modify to reject if detect any text
        return None

    def make_decision(self, count_results, text_count):
        """
        Make decision base on detections
        """
        decision = 0
        reason = ACCEPT_REASON
        return decision, reason

    def extract_max_mr_score(self, pred_matrix, det_preds):
        mr_indices = det_preds[:, -1] == 1  # MR class = 1
        mr_dets = det_preds[mr_indices]
        fake3d_subset = np.logical_and(det_preds[:, -1] == 4, det_preds[:, 4] > self.fake_threshold)
        fake3d_dets = det_preds[fake3d_subset] # Fake3D class = 4
        if len(mr_dets) == 0 or len(fake3d_dets) == 0:
            return np.max(pred_matrix[:, 1])
        
        overlaps = jaccard(mr_dets[:, :4], fake3d_dets[:, :4])
        # Only keep overlaps smaller than threshold
        true_mr_indices = (overlaps.sum(1) < self.fake_overlap_real).tolist()
        mr_matrix = pred_matrix[mr_indices][true_mr_indices]
        return np.max(mr_matrix[:, 1]) if len(mr_matrix) > 0 else 0.