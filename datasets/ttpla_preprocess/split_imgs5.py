import argparse
import os

output_folder = 'data/splitting_imgs'

train_folder = output_folder+'/'+'train'
val_folder = output_folder+'/'+'val'
test_folder = output_folder+'/'+'test'

ap = argparse.ArgumentParser()
ap.add_argument('-t', '--path_annotation_imgs', default="data/sized_data", help = 'path to jsons annotations')
args = ap.parse_args()

imgs_names = [js for js in os.listdir(args.path_annotation_imgs) if js.endswith(".jpg")]
train = []
with open('data/split_txts/train.txt','r') as hndl:
	for l in hndl:
		train.append(l.strip().replace(".json", ".jpg"))

test = []
with open('data/split_txts/test.txt','r') as hndl:
	for l in hndl:
		test.append(l.strip().replace(".json", ".jpg"))

val = []
with open('data/split_txts/val.txt','r') as hndl:
	for l in hndl:
		val.append(l.strip().replace(".json", ".jpg"))

print(len(train),len(test),len(val),len(imgs_names),len(train)+len(test)+len(val))
for p in [output_folder,train_folder,val_folder,test_folder]:
	if not os.path.exists(p):
		os.makedirs(p)

for t in imgs_names:
	if t in train:
		os.system('cp '+args.path_annotation_imgs+'/'+t+' '+train_folder+'/'+t)
	if t in test:
		os.system('cp '+args.path_annotation_imgs+'/'+t+' '+test_folder+'/'+t)
	if t in val:
		os.system('cp '+args.path_annotation_imgs+'/'+t+' '+val_folder+'/'+t)
