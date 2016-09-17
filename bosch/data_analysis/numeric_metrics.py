import csv

with open("../data/train_numeric.csv", "rt") as csvfile:
    reader = csv.reader(csvfile)
    rownum = 0
    for row in reader:
        if rownum == 0:
            headerRow = row
            cols = len(headerRow)
            counts = [0] * cols
            sums = [0] * cols
            squareSums = [0] * cols
        else:
            for x in range(cols):
                if row[x]:
                    counts[x] += 1
                    sums[x] += float(row[x])
                    squareSums[x] += float(row[x]) * float(row[x])
        rownum += 1
        if rownum > 100000:
            break
    for x in range(cols):
        if counts[x]:
            print headerRow[x], counts[x], sums[x] / counts[x], squareSums[x] / counts[x] - (sums[x] / counts[x]) * (sums[x] / counts[x])
