

import os, shutil, sys
from voc_eval import voc_eval_polygon
from collections import Counter
import numpy as np


eval_result_dir = "output/Analysis/output_eval"

# anno_path = 'data/ctw1500/test/test_label_curve.txt'

# outputstr = "dataset/ctw1500/Evaluation_sort/detections_text"


# score_thresh_list=[0.2, 0.3, 0.4, 0.5, 0.6, 0.62, 0.65, 0.7, 0.75, 0.8, 0.9]
score_thresh_list = [0.5]

anno_path = 'D:\\Text\\ctw1500-test-gt-new'
pred_file = 'D:\\Text\\Eval_Text\\pred_file.txt'


for iscore in score_thresh_list:
    rec, prec, AP, FP, TP, image_ids, num_gt = voc_eval_polygon(anno_path, pred_file, 'text', ovthresh=0.5)
    fid_path = '{}/Eval_ctw1500_{}.txt'.format(eval_result_dir, iscore)
    F = lambda x, y: 2 * x * y * 1.0 / (x + y)

    img_dict = dict(Counter(image_ids))

    with open(fid_path, 'w') as f:
        count = 0
        for k, v in zip(img_dict.keys(), img_dict.values()):
            fp = np.sum(FP[count:count+v])
            tp = np.sum(TP[count:count+v])
            count += v
            recall = tp / float(num_gt[k])
            precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            f.write('%s :: Precision=%.4f - Recall=%.4f\n' % (str(int(k)+1001)+".txt", recall, precision))

        Recall = rec[-1]
        Precision = prec[-1]
        F_score = F(Recall, Precision)
        f.write('ALL :: AP=%.4f - Precision=%.4f - Recall=%.4f - Fscore=%.4f' % (AP, Precision, Recall, F_score))

    print('AP: {:.4f}, recall: {:.4f}, pred: {:.4f}, '
          'FM: {:.4f}\n'.format(AP, Recall, Precision, F_score))

