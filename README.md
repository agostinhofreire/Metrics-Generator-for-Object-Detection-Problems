Metrics Generator for Object Detection Problems
====

Use this project to generate the metrics results for your object detection problem.

## **Running the Metrics Generator**

### **1st Step:**

Make sure you have installed all project dependencies, run in the root folder:

    pip install -r deploy/requirements.txt

The entire project uses `Python 3.6.13`.

### **2nd Step:**

You must provide two .txt files containing the dataset ground truth and the model's predictions informations. The files must be formatted with the following pattern.

`The ground truth file`

```
img1.jpg class_id xmin ymin xmax ymax class_id xmin ymin xmax ymax class_id xmin ymin xmax ymax ...
img2.jpg class_id xmin ymin xmax ymax class_id xmin ymin xmax ymax class_id xmin ymin xmax ymax ...
img3.jpg class_id xmin ymin xmax ymax class_id xmin ymin xmax ymax class_id xmin ymin xmax ymax ...
img4.jpg class_id xmin ymin xmax ymax class_id xmin ymin xmax ymax class_id xmin ymin xmax ymax ...
```
`The predictions file`

```
img1.jpg class_id confidence xmin ymin xmax ymax class_id confidence xmin ymin xmax ymax class_id confidence xmin ymin xmax ymax ...
img2.jpg class_id confidence xmin ymin xmax ymax class_id confidence xmin ymin xmax ymax class_id confidence xmin ymin xmax ymax ...
img3.jpg class_id confidence xmin ymin xmax ymax class_id confidence xmin ymin xmax ymax class_id confidence xmin ymin xmax ymax ...
img4.jpg class_id confidence xmin ymin xmax ymax class_id confidence xmin ymin xmax ymax class_id confidence xmin ymin xmax ymax ...

```
**Note:**

+ `class_id`: interger that represents a class in your dataset
+ `(xmin, ymin)`: bounding box top-left coordinate
+ `(xmax, ymax)`: bounding box bottom-right coordinate
+ `confidence`: prediction's confidence probability

### **3rd Step:**

Edit the Config.cfg file stored in the core folder.
```python
[MODEL]
classes = ['capacitor', 'diode', 'ic', 'inductor', 'resistor', 'transistor'] # your dataset classes
min_overlap = 0.5 # minimum overlap between bounding boxes
classes_to_ignore = [] # list of classes that you want to ignore

[RESULTS]
save_plots = True # save generated graphs
show_plots = False # plot generated graphs
dataset_name = "pcb-dataset" # dataset name

[DATA]
path_gt_file = "data/ground_truth.txt" # path to the ground truth file
path_pred_file = "data/predicted.txt" # path to the predictions file
```
### **4th Step:**

Execute de code by running:

    python main.py

Finally, all results will be saved in the root folder.
