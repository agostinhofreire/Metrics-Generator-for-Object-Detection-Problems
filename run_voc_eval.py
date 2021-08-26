from utils import voc_utils
from utils.utils import *
import pandas as pd
from configparser import ConfigParser
import ast

config_object = ConfigParser()

try:
  config_object.read("core/Config.cfg")
except:
  error("Config file not found in the core folder!")

try:

  model_config = config_object["MODEL"]

  classes = ast.literal_eval(model_config["classes"])
  min_overlap = ast.literal_eval(model_config["min_overlap"])

  results_config = config_object["RESULTS"]

  dataset_name = ast.literal_eval(results_config["dataset_name"])

  data_config = config_object["DATA"]

  path_gt_file = ast.literal_eval(data_config["path_gt_file"])
  path_pred_file = ast.literal_eval(data_config["path_pred_file"])

except:
  error("Verify the Config.cfg file!")

print("VOC Eval - {}".format(dataset_name))

preds = voc_utils.create_pred_list(path_pred_file)
gt = voc_utils.create_gt_dict(path_gt_file)

rec_total, prec_total, ap_total = voc_utils.AverageMeter(), voc_utils.AverageMeter(), voc_utils.AverageMeter()

results = {
  "Class": [],
  "Recall": [],
  "Precison": [],
  "AP": []
}

for ii in range(len(classes)):
  npos, nd, rec, prec, ap = voc_utils.voc_eval(gt, preds, ii, iou_thres=min_overlap)
  rec_total.update(rec, npos)
  prec_total.update(prec, nd)
  ap_total.update(ap, 1)

  results["Class"].append(classes[ii])
  results["Recall"].append(rec)
  results["Precison"].append(prec)
  results["AP"].append(ap)

  print('Class {}: Recall: {:.4f}, Precision: {:.4f}, AP: {:.4f}'.format(classes[ii], rec, prec, ap))

mAP = ap_total.avg
print('final mAP: {:.4f}'.format(mAP))
print("recall: {:.3f}, precision: {:.3f}".format(rec_total.avg, prec_total.avg))

df = pd.DataFrame.from_dict(results)
df.to_csv("{}.csv".format(dataset_name))