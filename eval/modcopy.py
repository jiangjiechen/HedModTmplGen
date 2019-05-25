import sys, os
from collections import defaultdict
import argparse
import annotate_desc

path = sys.path[0]
os.chdir(path)

parser = argparse.ArgumentParser()
parser.add_argument('-src', help='src-test file')
parser.add_argument('-res', help='result file')
args = parser.parse_args()

with open('stop_words_en.txt', 'r') as file:
    lines = file.readlines()
stop_word_lis = []
for line in lines:
    line = line.strip()
    stop_word_lis.append(line)

with open(args.src, 'r') as file:
    lines = file.readlines()

i = 0
value_lis_all = []
for line in lines:
    line = line.strip()
    triples = line.split(' ')
    lis = [ triple.split(u'￨')[0] for triple in triples ]
    lis = list(set(lis))
    value_lis_all.append( lis )

length = len(value_lis_all)

with open(args.res,'r') as file:
    lines = file.readlines()
desc_lis_all = []
for line in lines:
    line = line.strip()
    desc_lis_all.append( line.split(' ') )


copy_ratio_all = 0
skip_num = 0
for i in range(length):
    value_lis = value_lis_all[i]
    desc_lis = desc_lis_all[i]

    desc_line = ' '.join(desc_lis)
    if desc_line == '':
        skip_num += 1
        continue
    sent = annotate_desc.TypeDesc(desc_line)
    sent.find_heads_id()
    words = sent.words
    head_ids = sent.hed_ids
    heads = [ words[i] for i in head_ids ]

    for desc in desc_lis:
        if desc in heads:
            desc_lis.remove(desc)   # remove the ones that are heads

    for desc in desc_lis:
        if desc in stop_word_lis:
            desc_lis.remove(desc)   # remove the ones in stop words

    copy_num = 0
    all_num = len(desc_lis)
    
    for desc in desc_lis:
        for val in value_lis:
            if len(desc) > 4 and len(val) > 4:
                if desc[:4] == val[:4]:
                    copy_num += 1
                    continue
            if desc != '' and len(desc) <= 4 and len(val) > 4:
                if desc == val[:len(desc)]:
                    copy_num += 1
                    continue
            if len(desc) > 4 and len(val) <= 4 and val != '':
                if desc[:len(val)] == val:
                    copy_num += 1
                    continue
            if len(desc) <= 4 and len(val) <= 4 and desc != '' and val != '':
                mini = min(len(desc), len(val))
                if desc[:mini] == val[:mini]:
                    copy_num += 1
                    continue

    try:
        copy_ratio_all += copy_num/all_num
    except:
        skip_num += 1

avg_copy_ratio = copy_ratio_all/(length - skip_num)
print (skip_num)
print (length - skip_num)
print (avg_copy_ratio)

#java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000
#ps -ef | grep java | grep -v grep | awk '{print $2}' | sudo xargs kill -9
