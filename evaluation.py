import sys; sys.path.append('..')
from nlgeval import compute_metrics
import csv, os


def evaluate(res_file):
	if not os.path.exists(res_file): return {}
	print('*** Evaluating %s... ***' % res_file)

	file_name = res_file.split('.')[0]
	ref_file = file_name + '_ref.txt'
	hyp_file = file_name + '_hyp.txt'

	fout_1 = open(ref_file, 'w+')
	fout_2 = open(hyp_file, 'w+')
	with open(res_file, 'r') as f:
		for row in csv.reader(f, delimiter=';'):
			reference = row[0]
			hypothesis = row[1]
			fout_1.write(reference + '\n')
			fout_2.write(hypothesis + '\n')
	f.close()
	fout_1.close()
	fout_2.close()
	metrics_dict = compute_metrics(hypothesis=hyp_file, references=[ref_file])

	os.system('rm -rf %s' % ref_file)
	os.system('rm -rf %s' % hyp_file)

	return metrics_dict


if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("Please provide the eval_result_xxx.csv file as an argument.")
	elif len(sys.argv) == 2:
		res_file = sys.argv[1]
		evaluate(res_file)
	else:
		hyp_file = sys.argv[1]
		ref_file = sys.argv[2]
		metrics_dict = compute_metrics(hypothesis=hyp_file, references=[ref_file])
