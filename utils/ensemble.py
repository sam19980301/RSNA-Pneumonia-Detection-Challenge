""" 
Ensembling methods for object detection.
"""

""" 
General Ensemble - find overlapping boxes of the same class and average their positions
while adding their confidences. Can weigh different detectors with different weights.
No real learning here, although the weights and iou_thresh can be optimized.

Input: 
 - dets : List of detections. Each detection is all the output from one detector, and
          should be a list of boxes, where each box should be on the format 
          [box_x, box_y, box_w, box_h, class, confidence] where box_x and box_y 
          are the center coordinates, box_w and box_h are width and height resp.
          The values should be floats, except the class which should be an integer.

 - iou_thresh: Threshold in terms of IOU where two boxes are considered the same, 
               if they also belong to the same class.
               
 - weights: A list of weights, describing how much more some detectors should
            be trusted compared to others. The list should be as long as the
            number of detections. If this is set to None, then all detectors
            will be considered equally reliable. The sum of weights does not
            necessarily have to be 1.

Output:
    A list of boxes, on the same format as the input. Confidences are in range 0-1.
"""
import argparse
import csv
import os
from statistics import mean

submission_path = './submissions'

def GeneralEnsemble(dets, iou_thresh = 0.5, weights=None, avg_conf=None, conf_thresh=0.7):
    assert(type(iou_thresh) == float)
    
    ndets = len(dets)
    
    # weights of each model
    if weights is None:
        w = 1/float(ndets)
        weights = [w]*ndets
    else:
        assert(len(weights) == ndets)        
        # s = sum(weights)
        # for i in range(0, len(weights)):
        #     weights[i] /= s
    
    # average confidence of each model
    if avg_conf is None:
        avg_conf = 1/float(ndets)
    else:
        assert(len(avg_conf) == ndets)
        avg_conf = [a*w for a,w in zip(avg_conf, weights)]
        avg_conf = [s/sum(avg_conf) for s in avg_conf]

    out = list()
    used = list()
    
    for idet in range(0,ndets):
        det = dets[idet]
        for box in det:
            if box in used:
                continue
                
            used.append(box)
            # Search the other detectors for overlapping box of same class
            found = []
            for iodet in range(0, ndets):
                odet = dets[iodet]
                
                if odet == det:
                    continue
                
                bestbox = None
                bestiou = iou_thresh
                for obox in odet:
                    if not obox in used:
                        # Not already used
                        # print("class", box[4], obox[4])
                        # if box[4] == obox[4]:
                        # Same class
                        iou = computeIOU(box, obox)
                        if iou > bestiou:
                            bestiou = iou
                            bestbox = obox
                                
                if not bestbox is None:
                    w = weights[iodet]
                    found.append((bestbox,w, avg_conf[iodet]))
                    used.append(bestbox)
                            
            # Now we've gone through all other detectors
            if len(found) == 0:
                new_box = list(box)
                # conf = new_box[4] * weights[idet]
                new_box[0] = avg_conf[idet]
                if avg_conf[idet] > conf_thresh:
                    out.append(new_box)
            else:
                conf=0
                # store all boxes with similar IoU
                allboxes = [(box, weights[idet], avg_conf[idet])]
                allboxes.extend(found)
                # print('match box')
                # print(allboxes)
                
                xc = 0.0
                yc = 0.0
                bw = 0.0
                bh = 0.0
                conf = 0.0
                
                wsum = 0.0
                for bb in allboxes:
                    w = bb[1]
                    conf += bb[2] # reserve conf
                    wsum += w

                    b = bb[0]
                    # conf += w*b[0]
                    xc += w*b[1]
                    yc += w*b[2]
                    bw += w*b[3]
                    bh += w*b[4]
                
                xc /= wsum
                yc /= wsum
                bw /= wsum
                bh /= wsum

                # new_box = [xc, yc, bw, bh, conf]
                new_box = [conf, xc, yc, bw, bh]
                if conf > conf_thresh:
                    out.append(new_box)
            # print("add")
            # print(new_box)
    return out
    
def getCoords(box):
    x1 = float(box[0]) - float(box[2])/2
    x2 = float(box[0]) + float(box[2])/2
    y1 = float(box[1]) - float(box[3])/2
    y2 = float(box[1]) + float(box[3])/2
    return x1, x2, y1, y2
    
def computeIOU(box1, box2):
    x11, x12, y11, y12 = getCoords(box1[1:5])
    x21, x22, y21, y22 = getCoords(box2[1:5])
    
    x_left   = max(x11, x21)
    y_top    = max(y11, y21)
    x_right  = min(x12, x22)
    y_bottom = min(y12, y22)

    if x_right < x_left or y_bottom < y_top:
        return 0.0    
        
    intersect_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)        
    
    iou = intersect_area / (box1_area + box2_area - intersect_area)
    return iou

def read_info(patientInfo, csv_name, avg_conf):
    conf_sum = 0.0
    box_cnt = 0
    with open(csv_name, newline='') as csvfile:    
        head = True
        rows = csv.reader(csvfile)
        for (patientId, info) in rows:
            if head is True:
                head=False
                continue
            patientBboxes = []
            if patientId not in patientInfo.keys():
                patientInfo[patientId] = []
            split_info = info.split(' ')
            for i in range(int(len(split_info)/5)):
                conf, x, y, w, h = [float(i) for i in split_info[5*i:5*i+5]]
                conf_sum+=conf
                box_cnt+=1
                patientBboxes.append([conf, x, y, w, h])
            patientInfo[patientId].append(patientBboxes)
    
    # weight calculate reverse of average confidence
    avg_conf.append(float(conf_sum/box_cnt))
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, action="append", nargs="+")
    parser.add_argument("--output", type=str, default="submissions/ens_submission.csv")
    parser.add_argument("--thresh", type=float, default=None)
    args = parser.parse_args()

    return args

if __name__=="__main__":
    os.makedirs('submissions', exist_ok=True)
    
    args = parse_args()
    output_file_name = args.output
    patientInfo = {}
    avg_conf = []

    csv_files = []
    weights = []
    conf_thresh=args.thresh

    for (csv_file, weight) in args.csv:
        csv_files.append(csv_file)
        weights.append(float(weight))

    for csv_file in csv_files:
        read_info(patientInfo, csv_file, avg_conf)

    assert(len(avg_conf) == len(weights))

    # for i in range(len(weights)):
    #     weights[i] = weights[i]/avg_conf[i]
    weights = [w/sum(weights) for w in weights]
    # for i in range(len(weights)):
    #     conf_thresh += weights[i]*avg_conf[i]

    print("confidences:", avg_conf)
    print("weights:", weights)
    # print("confidence threshold:", conf_thresh)
    print("average confidence:", mean(avg_conf))

    if conf_thresh is None:
        conf_thresh = mean(avg_conf)
    # exit()

    if os.path.exists(output_file_name):
        os.remove(output_file_name)
    with open(output_file_name, 'a') as output_file:
        output_file.write('patientId,PredictionString\n')
        for key, info in patientInfo.items():
            line = key+','
            ens = GeneralEnsemble(info, weights=weights, avg_conf=avg_conf, conf_thresh=conf_thresh)
            # print(key)
            # print(ens)
            for bbox in ens:
                bbox = [str(i) for i in bbox]
                line += ' '.join(bbox) + ' '
            # print(line)
            output_file.write(line+'\n')

    print("output file", output_file_name)
        # res = GeneralEnsemble(patientInfo['c1937034-f8a4-4a84-a69c-213911b39907'], weights = weights)

    # print(patientInfo['c1937034-f8a4-4a84-a69c-213911b39907'])