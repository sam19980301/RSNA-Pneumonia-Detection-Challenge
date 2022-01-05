import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='./submission.csv', help="input submission file")
parser.add_argument('--shrink', type=float, default=0.875, help='shrinking rate of bboxes')
parser.add_argument('--thresh', type=float, default=0.0, help='threshold of confidence')
args = parser.parse_args()

submission_file = args.input
shrink_rate = args.shrink
conf_threshold = args.thresh


with open(submission_file, newline='') as csvfile:
    with open('new_submission.csv', 'a') as output_file:
        output_file.write('patientId,PredictionString\n')
        rows = csv.reader(csvfile)
        head=True
        for (patientId, info) in rows:
            if head is True:
                head=False
                continue
            line = ''
            line += patientId+','
            bboxes = []
            split_info = info.split(' ')
            for i in range(int(len(split_info)/5)):
                conf, x, y, w, h = [float(i) for i in split_info[5*i:5*i+5]]
                if conf <= conf_threshold:
                    continue

                shrink_x = x + w * (1-shrink_rate)/2
                shrink_y = y + h * (1-shrink_rate)/2
                shrink_w = w * shrink_rate
                shrink_h = h * shrink_rate
        
                bboxes += [str(i) for i in [
                    conf,
                    shrink_x,
                    shrink_y,
                    shrink_w,
                    shrink_h]]
            line += ' '.join(bboxes)
            print(line)
            output_file.write(line+'\n')

