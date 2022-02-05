import argparse

parser = argparse.ArgumentParser(description='Compute the average latency per frame.')
parser.add_argument('--file', dest='file', required=True, help='latency output file.')
args = parser.parse_args()

with open(args.file) as f:
    lats = f.readlines()

temp = 0
for lat in lats[1:]:
    temp += float(lat)

avg_lat = temp/(len(lats)-1)

print('Average Latency is {} ms'.format(avg_lat))
