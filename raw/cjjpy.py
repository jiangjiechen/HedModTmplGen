# -*- coding: utf-8 -*-

'''
@Author : Jiangjie Chen
@Time   : 2018/11/15 下午5:08
@Contact: mi0134sher@hotmail.com
'''

import re, datetime, json, random, os, sys, time
import urllib.request, urllib.parse, psutil
import argparse


def OverWriteCjjPy(root='.'):
	# import difflib
	# diff = difflib.HtmlDiff()
	cnt = 0
	golden_cjjpy = os.path.join(root, 'cjjpy.py')
	# golden_content = open(golden_cjjpy).readlines()
	for dir, folder, file in os.walk(root):
		for f in file:
			if f == 'cjjpy.py':
				cjjpy = '%s/%s' % (dir, f)
				# content = open(cjjpy).readlines()
				# d = diff.make_file(golden_content, content)
				cnt += 1
				print('[%d]: %s' % (cnt, cjjpy))
				os.system('cp %s %s' % (golden_cjjpy, cjjpy))


def ReplaceChar(file, replaced, replacer):
	print(file, replaced, replacer)
	with open(file) as f:
		data = f.readlines()
		out = open(file, 'w')
		for line in data:
			out.write(line.replace(replaced, replacer))


def DeUnicode(line):
	return line.encode('utf-8').decode('unicode_escape')


def LoadIDDict(dict_file):
	'''
	a\tb\n, `.dict' file
	'''
	assert dict_file.endswith('.dict')
	id2label = {}
	with open(dict_file, 'r') as f:
		data = f.read().split('\n')
		for i, line in enumerate(data):
			if line == '': continue
			try:
				id, label = line.split('\t')
				id2label[id] = label
			except:
				pass
	return id2label


def LoadInfobox(infobox_file, max_val_num=None):
	'''
	ent \t prop \t val；val \n`.infobox' file
	'''
	assert infobox_file.endswith('.infobox')
	infobox = {}
	with open(infobox_file) as f:
		for line in f:
			line = line.strip()
			if line == '': continue
			ent, prop, vals = line.split('\t')
			vals = [v for v in vals.split('|||')]
			if max_val_num:
				random.shuffle(vals)
				vals = vals[:max_val_num]
			if infobox.get(ent):
				infobox[ent][prop] = vals
			else:
				infobox[ent] = {prop: vals}
	return infobox


def LoadWords(file, is_file=True):
	if is_file:
		with open(file, 'r') as f:
			data = f.read().splitlines()
	else:
		data = file.splitlines()
	return set(map(lambda x: x.strip(), data))


def ChangeFileFormat(filename, new_fmt):
	assert type(filename) is str and type(new_fmt) is str
	spt = filename.split('.')
	if len(spt) == 0:
		return filename
	else:
		return filename.replace('.' + spt[-1], new_fmt)


class ShowProcess:
	"""
	Class to show progress bar
	"""
	i = 0  # where are we now
	max_steps = 0  # total steps
	max_arrow = 50  # bar length

	def __init__(self, max_steps):
		self.max_steps = max_steps
		self.i = 0
		self.st = time.time()

	# like [>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
	def show_process(self, i=None):
		now_time = time.time() - self.st
		if self.i == 0:
			remain_time = 0
		else:
			remain_time = now_time / (self.i / self.max_steps) - now_time
		if i is not None:
			self.i = i
		else:
			self.i += 1
		num_arrow = int(self.i * self.max_arrow / self.max_steps)  # how many '>'
		num_line = self.max_arrow - num_arrow  # how many '-'
		percent = self.i * 100.0 / self.max_steps  # progress xx.xx%
		# output string，'\r' returns to the left without new line
		process_bar = '[' + '>' * num_arrow + '-' * num_line + ']' \
		              + '%.2f' % percent + '%' + ' ETA %s' % (TimeClock(remain_time)) + '\r'
		sys.stdout.write(process_bar)
		sys.stdout.flush()

	def close(self, words='done'):
		print('')
		self.i = 0


def MakeDir(dir):
	if not os.path.exists(dir):
		os.makedirs(dir)


def GlanceFile(file, num=100):
	if num <= 0:
		cnt = 0
		for dir, folder, file in os.walk('.'):
			for f in file:
				if '_glance_' in f:
					cnt += 1
					filedir = os.path.join(dir, f)
					print('[%d]: Deleting %s...' % (cnt, filedir))
					os.system('rm %s' % filedir)
		return

	import bz2
	fmt = '.' + file.split('.')[-1]
	out = open(file.replace(fmt, '_glance_%d' % num + fmt), 'w')
	if file.endswith('.bz2'):
		f = bz2.open(file)
	else:
		f = open(file, 'r', encoding='utf-8')
	try:
		for i in range(num):
			line = f.readline()
			print(line)
			out.write(line)
		f.close()
	except Exception as e:
		print(e)
		print('That\'s the end of it.')
		f.close()


def SearchByKey(file, key):
	with open(file, 'r') as fin:
		while True:
			line = fin.readline()
			if not line: break
			if key in line:
				print(line, end='')


def SearchByKeys():
	cor = input('input a data source: ')
	while True:
		key = input('input a search key, -1 to end: ')
		if key == '-1':
			break
		SearchByKey(cor, key)


def ShowProgress(now, all, time, interval=10000):
	if now == 0:
		return
	remain_time = time / (now / all) - time
	if now % interval == 0:
		print('Progress at %f%%, cost %s, remaining %s' % ((now / all) * 100, TimeClock(time), TimeClock(remain_time)))


def GetDate():
	return str(datetime.datetime.now())[5:10].replace('-', '')


def TimeClock(seconds):
	sec = int(seconds)
	hour = int(sec / 3600)
	min = int((sec - hour * 3600) / 60)
	ssec = float(seconds) - hour * 3600 - min * 60
	return '%dh %dm %.2fs' % (hour, min, ssec)


def FileExists(filename):
	return os.path.isfile(filename)


def StripAll(text):
	return text.strip().replace('\t', '').replace('\n', '').replace(' ', '')


def GetBracket(text, bracket, en_br=False):
	# input should be aa(bb)cc, True for bracket, False for text
	if bracket:
		try:
			return re.findall('\（(.*?)\）', text.strip())[-1]
		except:
			return ''
	else:
		if en_br:
			text = re.sub('\(.*?\)', '', text.strip())
		return re.sub('（.*?）', '', text.strip())


def CharLang(uchar, lang):
	assert lang.lower() in ['en', 'cn']
	if lang.lower() == 'cn':
		if uchar >= '\u4e00' and uchar <= '\u9fa5':
			return True
		else:
			return False
	elif lang.lower() == 'en':
		if (uchar <= 'Z' and uchar >= 'A') or (uchar <= 'z' and uchar >= 'a'):
			return True
		else:
			return False
	else:
		return False


def WordLang(word, lang):
	for i in word.strip():
		if i.isspace(): continue
		if not CharLang(i, lang):
			return False
	return True


def SortDict(_dict, reverse=True):
	assert type(_dict) is dict
	return sorted(_dict.items(), key=lambda d: d[1], reverse=reverse)


def WriteIntoFile(data, write_file):
	try:
		if type(data) is not list:
			data = list(data)
	except:
		print('Data form error')
		return False

	out = open(write_file, 'wb')
	for line in data:
		out.write((line + '\n').encode('utf-8'))

	print('Writing completed.')
	return True


def MaxCommLen(str1, str2):
	lstr1 = len(str1)
	lstr2 = len(str2)
	record = [[0 for i in range(lstr2 + 1)] for j in range(lstr1 + 1)]
	max_num = 0
	for i in range(lstr1):
		for j in range(lstr2):
			if str1[i] == str2[j]:
				record[i + 1][j + 1] = record[i][j] + 1
				if record[i + 1][j + 1] > max_num:
					max_num = record[i + 1][j + 1]
	return max_num, ''


def TraceProgram(program):
	# TODO
	while True:
		time.sleep(5)
		is_running = False
		for p in psutil.process_iter():
			if program == p.name():
				is_running = True
		if not is_running:
			SendEmail('%s exits.' % program, 'From %s' % socket.gethostname())
			break


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--diff', nargs=2,
	                    help='show difference between two files, shown in downloads/diff.html')
	parser.add_argument('--de_unicode', action='store_true', default=False,
	                    help='remove unicode characters')
	parser.add_argument('--eval_isa', action='store_true', default=False,
	                    help='')
	parser.add_argument('--found_mention_in_cndb', action='store_true', default=False,
	                    help='')
	parser.add_argument('--glance', nargs=2,
	                    help='file glancer, 2 args: file name & number of glance lines')
	parser.add_argument('--link_entity', action='store_true', default=False,
	                    help='')
	parser.add_argument('--make_dir', action='store_true', default=False,
	                    help='')
	parser.add_argument('--max_comm_len', action='store_true', default=False,
	                    help='')
	parser.add_argument('--mention2entity', action='store_true', default=False,
	                    help='')
	parser.add_argument('--search', nargs=2,
	                    help='search key from file, 2 args: file name & key')
	parser.add_argument('--email', action='store_true', default=False,
	                    help='')
	parser.add_argument('--trace_program', action='store_true', default=False,
	                    help='')
	parser.add_argument('--overwrite', action='store_true', default=None,
	                    help='overwrite all cjjpy under given *dir* based on *dir*/cjjpy.py')
	parser.add_argument('--replace', nargs=3,
	                    help='replace char, 3 args: file name & replaced char & replacer char')
	args = parser.parse_args()

	if args.glance:
		print('* Glancing File...')
		fst = args.glance[0]
		sec = args.glance[1]
		file = sec if fst.isdigit() else fst
		num = fst if fst.isdigit() else sec
		GlanceFile(file, int(num))

	if args.overwrite:
		print('* Overwriting cjjpy...')
		OverWriteCjjPy()

	if args.replace:
		print('* Replacing Char...')
		ReplaceChar(args.replace[0], args.replace[1], args.replace[2])

	if args.search:
		file = args.search[0]
		key = args.search[1]
		print('* Searching %s from %s...' % (key, file))
		SearchByKey(file, key)
