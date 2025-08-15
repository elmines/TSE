#!/usr/bin/env python3
import csv
import sys

reader = csv.DictReader(sys.stdin)

writer = csv.DictWriter(sys.stdout, fieldnames=reader.fieldnames + ['GT Target', 'Mapped Target'])
writer.writeheader()

for row in reader:
    row['Mapped Target'] = row['GT Target'] = row['Target']
    writer.writerow(row)