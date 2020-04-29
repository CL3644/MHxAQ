import csv 
import sys
import operator
from uszipcode import SearchEngine, SimpleZipcode, Zipcode
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from random import seed
from random import random


with open(sys.argv[1], 'r') as raw:
	rawData = list(csv.reader(raw, delimiter=","))

filtered = []
filters = [2, 17, 18, 19, 20, 32, 40, 47, 66, 4, 3, 13, 12]

filtered = map(operator.itemgetter(*filters), rawData)
filtered = [list(elem) for elem in filtered]

for row in filtered:
	for entry in range(len(row)):
		if row[entry] == "YES":
			row[entry] = 1
		elif row[entry] == "NO":
			row[entry] = 0
		elif "UNKNOWN" in row[entry]:
			row[entry] = 0
		elif row[entry] == "FEMALE":
			row[entry] = 0
		elif row[entry] == "MALE":
			row[entry] = 1
		elif row[entry] == "ADULT":
			row[entry] = 1
		elif row[entry] == "CHILD":
			row[entry] = 0
		elif "LOOKING" in row[entry]:
			row[entry] = 0
		elif "VOLUN" in row[entry]:
			row[entry] = 0
		elif row[entry] == "EMPLOYED":
			row[entry] = 1

nyc_count = 0

d2 = []
threedigits = {}

for row in filtered:
	if row[0] == "NEW YORK CITY REGION" or row[0] == "Region Served":
		if row[8] != "888" and row[8] != "999":
			nyc_count += 1
 			d2.append(row[1:])
 			if row[8] in threedigits.keys():
 				threedigits[row[8]] += 1
 			else:
 				threedigits[row[8]] = 1

toremove = []
toremove.append("115")

conversions= {'114': '110', '117': '103', '116':'112', '111':'110', '113':'110'}

for key in  threedigits.keys():
	if threedigits[key] < 150:
		toremove.append(key)
		threedigits.pop(key)



zipcount = {}

for row in d2:
	if row[7] in conversions.keys():
		row[7] = conversions[row[7]]
	if row[7] == '101':
		row[7] == '100'
	if row[7] in zipcount.keys():
		zipcount[row[7]] +=1
	else:
		zipcount[row[7]] = 1

y = [s for s in d2 if (s[7] == "Three Digit Residence Zip Code" or s[7] not in  toremove)]
y = [s for s in y if s[7] != '101']

print(y)

#check  y ffor unwanted zip codes 

check = {}
for row in y:
	if row[7] in check.keys():
		check[row[7]] += 1
	else:
		check[row[7]] = 1

print(check)



print("entries from NYC: " + str(nyc_count))

print(d2[:3])

with open(sys.argv[2], 'r') as raw2:
	aq = list(csv.reader(raw2, delimiter = ","))

columns = aq[0]
filtered_aq = []

x = 0
for col in columns:
	print("[" + str(x) + "] " + col)
	x+=1

boroughs = {}
formatted = {}

for row in aq:
	if row[4] == "Borough" and "Rate" not in row[3] and "km2" not in row[3]:
		filtered_aq.append(row[2:])
		if row[6] in boroughs.keys():
			boroughs[row[6]] += 1
		else:
			boroughs[row[6]] = 1
		
# print(boroughs)

for b in boroughs.keys():
	formatted[b] = []

print(filtered_aq[:5])

for row in filtered_aq:
	if row[4] in formatted.keys():
		formatted[row[4]].append((row[0] + "::" + row[1], row[6]))

print(formatted)

for tup in formatted["Queens"]:
	d2[0].append(tup[0])

for key in formatted.keys():
	if key == "Bronx":
		formatted["104"] = formatted.pop("Bronx")
	if key == "Brooklyn":
		formatted["112"] = formatted.pop("Brooklyn")
	if key == "Staten Island":
		formatted["103"] = formatted.pop("Staten Island")
	if key == "Manhattan":
		formatted["100"] = formatted.pop("Manhattan")
	if key == "Queens":
		formatted["110"] = formatted.pop("Queens")

mh_count = 0
entries = 0

counts = {}

for x in range(0, 8):
	counts[y[0][x]] = 0

y[0].append("random value")

seed(1)

for row in range(1,len(y)):	
	entries += 1
	iterator = 0
	for d in counts.keys():
		if y[row][iterator] == 1:
			counts[d] +=1
		iterator+=1
	if y[row][0] == 1:
		mh_count+=1
	for metric in d2[0]:
		for tup in formatted[y[row][7]]:
			if metric == tup[0]:
				y[row].append(tup[1])
	y[row].append(random())


print(y[:10])
print("mh_count: " + str(mh_count))
print("enties: " + str(entries))
print(counts)






with open('aq_filtered.csv', mode='w') as aq_edited:
    aq_writer = csv.writer(aq_edited, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for row in filtered_aq:
		aq_writer.writerow(row)

with open('mh_filtered.csv', mode='w') as mh_edited:
    mh_writer = csv.writer(mh_edited, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for row in y:
		mh_writer.writerow(row)

with open('final_head.csv', mode='w') as head:
    head_writer = csv.writer(head, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for row in y[:100]:
		head_writer.writerow(row)

with open('data.csv', mode='w') as yy:
    y_write = csv.writer(yy, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for row in y:
    	del row[6]
    	y_write.writerow(row)

################################################################################################################################################









