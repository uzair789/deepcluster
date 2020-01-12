import matplotlib.pyplot as plt
import os



folders = ['results/exp_uz_l2_K1000_debug',
		'results/exp_uz_l2_K10000_debug',
		'results/exp_uz_l2_K100000_debug',
		'results/exp_uz_l2_K5950_debug',
		'results/exp_uz_l2_K59500_debug']


def get_nmi_values(log_file_path):

	f = open(log_file_path, 'r')
	nmi = []

	for line in f:
		line = line.strip()
		v = line.split('=  ')
		val = round( float(v[-1]), 3)
		nmi.append(val)
	return nmi
	

fig, ax  = plt.subplots()

for folder in folders:
	log_file_path = os.path.join(folder, 'log.txt')
	nmi = get_nmi_values(log_file_path)
	suffix = folder.split('_')[3]
	x = range(len(nmi))
	ax.plot(x, nmi, label=suffix)

ax.legend(loc='lower right')



plt.savefig('master_nmi.jpg')
