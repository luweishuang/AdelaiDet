import json
import argparse
import os

use_cats = ["cable"]

new_jsons = 'data/newjsons'
for p in [new_jsons]:
	if not os.path.exists(p):
		os.makedirs(p)


ap = argparse.ArgumentParser()
ap.add_argument('-t', '--path_annotation_jsons', default="data/sized_data", help = 'path to imgs with annotations')
args = ap.parse_args()

jsons_names = [img for img in os.listdir(args.path_annotation_jsons) if img.endswith(".json")]
all_labels = {}
for js in jsons_names:
	fpath = args.path_annotation_jsons+'/'+js
	fdata = json.load(open(fpath,'r'))
	newdata = {}
	nvoids = 0
	for key in fdata:
		if key == 'shapes':
			newdata['shapes'] = []
			for m in fdata['shapes']:
				m['label'] = m['label'].lower()
				if m['label'] == 'void':
					nvoids += 1
					continue
				elif m['label'] not in use_cats:
					nvoids += 1
					continue
				newdata['shapes'].append(m)
		else:
			newdata[key] = fdata[key]
	print('remove %d voids from %s'%(nvoids,js)+' and saving the new file to '+new_jsons+'/'+js)
	json.dump(newdata,open(new_jsons+'/'+js,'w'))
