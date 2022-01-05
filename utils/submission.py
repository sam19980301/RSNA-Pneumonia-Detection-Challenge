import os
import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_anno', type=str, default="../RSNA_answer.json", help="image annotations for submission in coco format")
parser.add_argument('--infer_anno', type=str, default='./inference.bbox.json', help='''inference result for submission in coco format, \
    e.g. \
    /tools/dist_test.sh \
    config.py \
    checkpoint.pth \
    1 \
    --format-only \
    --options "jsonfile_prefix=./inference"
    ''')
parser.add_argument('--output', type=str, default='./submission.csv', help="output submission filename in csv format")
args = parser.parse_args()

img_anno = args.img_anno
infer_anno = args.infer_anno
output = args.output

# load inference json
with open(img_anno,'r') as f:
    img_id_map = json.load(f)
    img_id_map = img_id_map['images']
    img_id_map = {i['id']:i['file_name'].rstrip('.png') for i in img_id_map}

with open(infer_anno,'r') as f:
    pred = json.load(f)
    pred = pd.DataFrame(pred).drop(columns=['category_id'])                                                             

# submission formatter
def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        if j[0] < 0.0:
            continue
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], 
                                                             int(j[1][0]), int(j[1][1]), 
                                                             int(j[1][2]), int(j[1][3])))
    return " ".join(pred_strings) if len(pred_strings)>0 else None

# generate submission file
result = pd.DataFrame(index=range(len(img_id_map)),columns=['patientId', 'PredictionString'])
for i in range(len(img_id_map)):
    single_pred = pred[pred['image_id']== i][['bbox','score']]
    result.loc[i,'patientId'] = img_id_map[i]
    result.loc[i,'PredictionString'] = format_prediction_string(single_pred['bbox'],single_pred['score'])
result.to_csv(output, index=False)
