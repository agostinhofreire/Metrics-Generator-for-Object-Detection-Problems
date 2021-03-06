import os
import operator
import sys

import cv2
import matplotlib.pyplot as plt
from xml.dom import minidom

def get_xml_infos(path_xml_file):
    root = minidom.parse(path_xml_file)

    try:

        objects = root.getElementsByTagName("object")

        size = root.getElementsByTagName("size")[0]
        width = int(float(size.getElementsByTagName("width")[0].firstChild.nodeValue))
        height = int(float(size.getElementsByTagName("height")[0].firstChild.nodeValue))
        depth = int(float(size.getElementsByTagName("depth")[0].firstChild.nodeValue))

    except:

        return [], None, None, None

    objects_list = []  # (xmin, ymin, xmax, ymax)
    for obj in objects:

        try:
            bndbox = obj.getElementsByTagName("bndbox")[0]
            name = str(obj.getElementsByTagName("name")[0].firstChild.data)
            name = name.lower()


            xmin = int(float(bndbox.getElementsByTagName("xmin")[0].firstChild.nodeValue))
            ymin = int(float(bndbox.getElementsByTagName("ymin")[0].firstChild.nodeValue))
            xmax = int(float(bndbox.getElementsByTagName("xmax")[0].firstChild.nodeValue))
            ymax = int(float(bndbox.getElementsByTagName("ymax")[0].firstChild.nodeValue))
            objects_list.append((name, xmin, ymin, xmax, ymax))

        except:

            continue

    return objects_list, width, height, depth

def error(msg):
  print(msg)
  sys.exit(0)

def split_objects(obj_list, steps):
    new_obj_list = []
    temp_list = []
    for item in obj_list:
        
        temp_list.append(item)

        if len(temp_list) >= steps:
            new_obj_list.append(temp_list)
            temp_list = []

    return new_obj_list


def create_p_gt_folders(path_gt_file, path_pred_file, name, classes):
    gt = open(path_gt_file, "r")

    if not os.path.exists("data/"):
            os.makedirs("data/")
    
    gt_folder_name = os.path.join("data/", f"ground-truth-{name}")
    p_folder_name = os.path.join("data/", f"predicted-{name}")

    for folder in [gt_folder_name, p_folder_name]:
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            if len(os.listdir(folder)) > 0:
                error(f"Check the {folder} folder, it already exists and it is not empty!")

    gt_lines = gt.readlines()
    gt.close()

    for gt_line in gt_lines:
        temp_gt_line = gt_line.replace("\n", "")

        if temp_gt_line:
                
            temp_gt_line = temp_gt_line.split()

            img_name = temp_gt_line[0].split(".")[0]
            path_file = os.path.join(gt_folder_name, img_name + ".txt")
            path_p_file = os.path.join(p_folder_name, img_name + ".txt")
            temp_gt_line = temp_gt_line[1:]
            temp_gt_line = split_objects(temp_gt_line, 5)
            new_lines = []
            for class_id, xmin, ymin, xmax, ymax in temp_gt_line:
                class_name = classes[int(class_id)]
                line = f"{class_name} {xmin} {ymin} {xmax} {ymax}\n"
                new_lines.append(line)

            temp_file = open(path_file, "w+")
            temp_file.writelines(new_lines)
            temp_file.close()

            temp_p_file = open(path_p_file, "w+")
            temp_p_file.close()


    
    p = open(path_pred_file, "r")
    p_lines = p.readlines()
    p.close()

    for p_line in p_lines:
        temp_p_line = p_line.replace("\n", "")

        if temp_p_line:

            temp_p_line = temp_p_line.split()

            img_name = temp_p_line[0].split(".")[0]
            path_file = os.path.join(p_folder_name, img_name + ".txt")
            temp_p_line = temp_p_line[1:]
            temp_p_line = split_objects(temp_p_line, 6)
            new_lines = []
            for class_id, conf, xmin, ymin, xmax, ymax in temp_p_line:
                class_name = classes[int(class_id)]
                line = f"{class_name} {conf} {xmin} {ymin} {xmax} {ymax}\n"
                new_lines.append(line)

            temp_file = open(path_file, "w+")
            temp_file.writelines(new_lines)
            temp_file.close()

    files_pred = os.listdir(p_folder_name)
    files_gt = os.listdir(gt_folder_name)

    for f_pred in files_pred:
        if f_pred not in files_gt:
            path = os.path.join(gt_folder_name, f_pred)
            file = open(path, "w+")
            file.write("")
            file.close()


"""
 check if the number is a float between 0.0 and 1.0
"""
def is_float_between_0_and_1(value):
  try:
    val = float(value)
    if val > 0.0 and val < 1.0:
      return True
    else:
      return False
  except ValueError:
    return False

"""
 Calculate the AP given the recall and precision array
  1st) We compute a version of the measured precision/recall curve with
       precision monotonically decreasing
  2nd) We compute the AP as the area under this curve by numerical integration.
"""
def voc_ap(rec, prec):
  """
  --- Official matlab code VOC2012---
  mrec=[0 ; rec ; 1];
  mpre=[0 ; prec ; 0];
  for i=numel(mpre)-1:-1:1
      mpre(i)=max(mpre(i),mpre(i+1));
  end
  i=find(mrec(2:end)~=mrec(1:end-1))+1;
  ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
  """
  rec.insert(0, 0.0) # insert 0.0 at begining of list
  rec.append(1.0) # insert 1.0 at end of list
  mrec = rec[:]
  prec.insert(0, 0.0) # insert 0.0 at begining of list
  prec.append(0.0) # insert 0.0 at end of list
  mpre = prec[:]
  """
   This part makes the precision monotonically decreasing
    (goes from the end to the beginning)
    matlab:  for i=numel(mpre)-1:-1:1
                mpre(i)=max(mpre(i),mpre(i+1));
  """
  # matlab indexes start in 1 but python in 0, so I have to do:
  #   range(start=(len(mpre) - 2), end=0, step=-1)
  # also the python function range excludes the end, resulting in:
  #   range(start=(len(mpre) - 2), end=-1, step=-1)
  for i in range(len(mpre)-2, -1, -1):
    mpre[i] = max(mpre[i], mpre[i+1])
  """
   This part creates a list of indexes where the recall changes
    matlab:  i=find(mrec(2:end)~=mrec(1:end-1))+1;
  """
  i_list = []
  for i in range(1, len(mrec)):
    if mrec[i] != mrec[i-1]:
      i_list.append(i) # if it was matlab would be i + 1
  """
   The Average Precision (AP) is the area under the curve
    (numerical integration)
    matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
  """
  ap = 0.0
  for i in i_list:
    ap += ((mrec[i]-mrec[i-1])*mpre[i])
  return ap, mrec, mpre


"""
 Convert the lines of a file to a list
"""
def file_lines_to_list(path):
  # open txt file lines to a list
  with open(path) as f:
    content = f.readlines()
  # remove whitespace characters like `\n` at the end of each line
  content = [x.strip() for x in content]
  return content

"""
 Draws text in image
"""
def draw_text_in_image(img, text, pos, color, line_width):
  font = cv2.FONT_HERSHEY_PLAIN
  fontScale = 1
  lineType = 1
  bottomLeftCornerOfText = pos
  cv2.putText(img, text,
      bottomLeftCornerOfText,
      font,
      fontScale,
      color,
      lineType)
  text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
  return img, (line_width + text_width)

"""
 Plot - adjust axes
"""
def adjust_axes(r, t, fig, axes):
  # get text width for re-scaling
  bb = t.get_window_extent(renderer=r)
  text_width_inches = bb.width / fig.dpi
  # get axis width in inches
  current_fig_width = fig.get_figwidth()
  new_fig_width = current_fig_width + text_width_inches
  propotion = new_fig_width / current_fig_width
  # get axis limit
  x_lim = axes.get_xlim()
  axes.set_xlim([x_lim[0], x_lim[1]*propotion])

"""
 Draw plot using Matplotlib
"""
def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, show_plots, plot_color, true_p_bar):
  # sort the dictionary by decreasing value, into a list of tuples
  sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
  # unpacking the list of tuples into two lists
  sorted_keys, sorted_values = zip(*sorted_dic_by_value)
  # 
  if true_p_bar != "":
    """
     Special case to draw in (green=true predictions) & (red=false predictions)
    """
    fp_sorted = []
    tp_sorted = []
    for key in sorted_keys:
      fp_sorted.append(dictionary[key] - true_p_bar[key])
      tp_sorted.append(true_p_bar[key])
    plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Predictions')
    plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Predictions', left=fp_sorted)
    # add legend
    plt.legend(loc='lower right')
    """
     Write number on side of bar
    """
    fig = plt.gcf() # gcf - get current figure
    axes = plt.gca()
    r = fig.canvas.get_renderer()
    for i, val in enumerate(sorted_values):
      fp_val = fp_sorted[i]
      tp_val = tp_sorted[i]
      fp_str_val = " " + str(fp_val)
      tp_str_val = fp_str_val + " " + str(tp_val)
      # trick to paint multicolor with offset:
      #   first paint everything and then repaint the first number
      t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
      plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
      if i == (len(sorted_values)-1): # largest bar
        adjust_axes(r, t, fig, axes)
  else:
    plt.barh(range(n_classes), sorted_values, color=plot_color)
    """
     Write number on side of bar
    """
    fig = plt.gcf() # gcf - get current figure
    axes = plt.gca()
    r = fig.canvas.get_renderer()
    for i, val in enumerate(sorted_values):
      str_val = " " + str(val) # add a space before
      if val < 1.0:
        str_val = " {0:.2f}".format(val)
      t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
      # re-set axes to show number inside the figure
      if i == (len(sorted_values)-1): # largest bar
        adjust_axes(r, t, fig, axes)
  # set window title
  fig.canvas.set_window_title(window_title)
  # write classes in y axis
  tick_font_size = 12
  plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
  """
   Re-scale height accordingly
  """
  init_height = fig.get_figheight()
  # comput the matrix height in points and inches
  dpi = fig.dpi
  height_pt = n_classes * (tick_font_size * 1.4) # 1.4 (some spacing)
  height_in = height_pt / dpi
  # compute the required figure height 
  top_margin = 0.15    # in percentage of the figure height
  bottom_margin = 0.05 # in percentage of the figure height
  figure_height = height_in / (1 - top_margin - bottom_margin)
  # set new height
  if figure_height > init_height:
    fig.set_figheight(figure_height)

  # set plot title
  plt.title(plot_title, fontsize=14)
  # set axis titles
  # plt.xlabel('classes')
  plt.xlabel(x_label, fontsize='large')
  # adjust size of window
  fig.tight_layout()
  # save the plot
  fig.savefig(output_path)
  # show image
  if show_plots:
    plt.show()
  # close the plot
  plt.close()