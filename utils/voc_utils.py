import sys

import numpy as np

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.average = self.sum / float(self.count)

    @property
    def avg(self):
        self.last_avg = self.average
        self.reset()
        return self.last_avg

def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval(gt_dict, val_preds, classidx, iou_thres=0.5, use_07_metric=False):
    '''
    Top level function that does the PASCAL VOC evaluation.
    '''
    # 1.obtain gt: extract all gt objects for this class
    class_recs = {}
    npos = 0
    for img_id in gt_dict:
        R = [obj for obj in gt_dict[img_id] if obj[-1] == classidx]
        bbox = np.array([x[:4] for x in R])
        det = [False] * len(R)
        npos += len(R)
        class_recs[img_id] = {'bbox': bbox, 'det': det}

    # 2. obtain pred results
    pred = [x for x in val_preds if x[-1] == classidx]
    img_ids = [x[0] for x in pred]
    confidence = np.array([x[-2] for x in pred])
    BB = np.array([[x[1], x[2], x[3], x[4]] for x in pred])

    # 3. sort by confidence
    sorted_ind = np.argsort(-confidence)
    try:
        BB = BB[sorted_ind, :]
    except:
        return 1e-6, 1e-6, 0, 0, 0
    img_ids = [img_ids[x] for x in sorted_ind]

    # 4. mark TPs and FPs
    nd = len(img_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        # all the gt info in some image
        try:
            R = class_recs[img_ids[d]]
        except:
            R = {'bbox': np.array([]), 'det': np.array([])}
        bb = BB[d, :]
        ovmax = -np.Inf
        BBGT = R['bbox']

        if BBGT.size > 0:
            # calc iou
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + (BBGT[:, 2] - BBGT[:, 0] + 1.) * (
                        BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > iou_thres:
            # gt not matched yet
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    # return rec, prec, ap
    return npos, nd, tp[-1] / float(npos), tp[-1] / float(nd), ap

def line_parser_gt(line):
    line_elements = line.split(" ")

    image = line_elements[0]
    line_elements = line_elements[1:]
    box_count = len(line_elements)//5

    boxes = []
    labels = []

    for i in range(box_count):
        label, x_min, y_min, x_max, y_max = int(line_elements[i * 5]), int(line_elements[i * 5 + 1]), int(line_elements[i * 5 + 2]), int(
            line_elements[i * 5 + 3]), int(line_elements[i * 5 + 4])
        boxes.append([x_min, y_min, x_max, y_max])
        labels.append(label)

    boxes = np.asarray(boxes, np.float32)
    labels = np.asarray(labels, np.int64)

    return image, boxes, labels

def create_gt_dict(gt_file_path):
    gt = {}
    with open(gt_file_path, "r") as file:
        for line in file:
            image, boxes, labels = line_parser_gt(line)

            objects = []
            for i in range(len(labels)):
                x_min, y_min, x_max, y_max = boxes[i]
                label = labels[i]
                objects.append([x_min, y_min, x_max, y_max, label])
            gt[image] = objects

    return gt

def line_parser_pred(line):

    line_elements = line.split(" ")

    image = line_elements[0]
    line_elements = line_elements[1:]
    box_count = len(line_elements)//6

    preds = []

    for i in range(box_count):
        label, score, x_min, y_min, x_max, y_max = int(line_elements[i * 6]), float(line_elements[i * 6 + 1]), int(line_elements[i * 6 + 2]), int(
            line_elements[i * 6 + 3]), int(line_elements[i * 6 + 4]), int(line_elements[i * 6 + 5])

        preds.append([image, x_min, y_min, x_max, y_max, score, label])

    return preds


def create_pred_list(pred_file_path):
    pred = []
    with open(pred_file_path, "r") as file:
        for line in file:

            img_pred = line_parser_pred(line)
            pred.extend(img_pred)

    return pred

