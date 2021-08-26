from utils.utils import *
import os
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
  classes_to_ignore = ast.literal_eval(model_config["classes_to_ignore"])

  data_config = config_object["DATA"]

  path_gt_file = ast.literal_eval(data_config["path_gt_file"])
  path_gt_xml_files = ast.literal_eval(data_config["path_gt_xml_files"])
  classes_to_rename = ast.literal_eval(data_config["classes_to_rename"])

except:
  error("Verify the Config.cfg file!")

try:
  annotations = os.listdir(path_gt_xml_files)
except:
  error("Invalid folder path!")

content = ""

annotations.sort()

for annotation in annotations:

  image = annotation.replace(".xml", ".jpg")

  line = "\n{}".format(image)

  path_annotation = os.path.join(path_gt_xml_files, annotation)

  objects_list, width, height, depth = get_xml_infos(path_annotation)

  if objects_list:
    for name, xmin, ymin, xmax, ymax in objects_list:

      if name in classes_to_ignore:
        continue

      if name in classes_to_rename.keys():
        class_id = classes.index(classes_to_rename[name])

      else:
        class_id = classes.index(name)

      bboxes = f" {class_id} {xmin} {ymin} {xmax} {ymax}"

      line += bboxes

    content += line

content = content.replace("\n", "", 1)

with open(path_gt_file, "w+") as file:
  file.write(content)


