#!/usr/bin/env python3
from sklearn.metrics import precision_recall_fscore_support
from collections import OrderedDict
import csv
import sys

slices = OrderedDict([
    ('SE', slice(0, 1080)),
    ('AM', slice(1880, 6989)),
    ('C19', slice(1080, 1880)),
    ('PS', slice(6989, 9146)),
    ('All', slice(None))
])

writer = csv.DictWriter(sys.stdout, fieldnames=list(slices.keys()))
writer.writeheader()

result_rows = []
for in_path in sys.argv[1:]:
    pred_strs = []
    label_strs = []
    targets = set()
    with open(in_path, 'r') as r:
        reader = csv.DictReader(r)
        for row in reader:
            pred_strs.append(row['Mapped Target'])
            gt_target = row['Target'] if 'Target' in row else row['GT Target']
            label_strs.append(gt_target)
            targets.add(gt_target)
    target2id = {t:i for i,t in enumerate(targets)}

    preds = [target2id[t] for t in pred_strs]
    labels = [target2id[t] for t in label_strs]

    row = {}
    for dataset_name, inds in slices.items():
        d_preds = preds[inds]
        d_labels = labels[inds]
        _, _, micro_f1, _ = precision_recall_fscore_support(d_labels, d_preds, average='micro')
        row[dataset_name] = micro_f1
    result_rows.append(row)
    writer.writerow(row)

means = {k:sum(row[k] for row in result_rows)/len(result_rows) for k in slices}
writer.writerow(means)