import sys
import os, argparse
import annotate_desc
from collections import defaultdict

path = sys.path[0]
os.chdir(path)

parser = argparse.ArgumentParser()
parser.add_argument('-src', help='src-test file')
parser.add_argument('-tgt', help='tgt-test file')
parser.add_argument('-res', help='result file')
args = parser.parse_args()

ground_truth_lis_all = []
with open(args.tgt,'r') as file:
    lines = file.readlines()
for line in lines:
    line = line.strip()
    sent = annotate_desc.TypeDesc(line)
    sent.find_heads_id()
    words = sent.words
    head_ids = sent.hed_ids
    heads = [ words[i] for i in head_ids ]
    ground_truth_lis_all.append( heads )

sub_is_list_all = []
length = len(ground_truth_lis_all)
with open(args.src,'r') as file:
    lines = file.readlines()
for line in lines:
    line = line.strip()
    triples = line.split(' ')
    lis = []
    for triple in triples:
        if 'instance_of' in triple:
            val, _, _ = triple.split(u'￨')
            lis.append(val)
        if 'subclass_of' in triple:
            val, _, _ = triple.split(u'￨')
            lis.append(val)
    lis = list(set(lis))
    sub_is_list_all.append(lis)


all_num = 0
correct_num = 0
with open(args.res,'r') as file:    #change
    lines = file.readlines()
i = 0
for line in lines:
    line = line.strip()
    if line == '':
        i += 1
        continue
    sent = annotate_desc.TypeDesc(line)
    sent.find_heads_id()
    words = sent.words
    head_ids = sent.hed_ids
    heads = [ words[i] for i in head_ids ]

    all_num += len(heads)

    gt_lis = ground_truth_lis_all[i]
    si_lis = sub_is_list_all[i]
    for head in heads:
        if head in gt_lis or head in si_lis:
            #print (head, gt_lis, si_lis)
            correct_num += 1
        else:
            #print (head, gt_lis, si_lis)
            pass
    i += 1

print (correct_num/all_num)
#java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000
#ps -ef | grep java | grep -v grep | awk '{print $2}' | sudo xargs kill -9