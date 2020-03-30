import matplotlib.pyplot as plt
import os



folders = ['../exp_uz_resnet_K1000_seqFix', '../results/exp']



def get_values(log_file_path):

	f = open(log_file_path, 'r')
	nmi = []
	conv_loss = []
	cluster_loss  = []
	for line in f:
		line = line.rstrip()
		if '###### Epoch' in line:
			print(line)
		if 'Clustering loss:' in line:
			clus_loss_val = line.split('Clustering loss: ')[1]
			cluster_loss.append(float(clus_loss_val))
		if 'ConvNet loss:' in line:
			conv_loss_val = line.split('ConvNet loss: ')[1]
			conv_loss.append(float(conv_loss_val))
		if 'NMI against previous assignment:' in line:
			nmi_val = line.split('NMI against previous assignment: ')[1]
			nmi.append(float(nmi_val))
		#nmi.append(val)

	print(cluster_loss)
	print('--')
	print(conv_loss)
	print('--')
	print(nmi)
	return nmi, conv_loss, cluster_loss
	

fig1, ax1  = plt.subplots()
fig2, ax2  = plt.subplots()
fig3, ax3  = plt.subplots()

for folder in folders:
	log_file_path = os.path.join(folder, 'log.txt')
	[nmi, conv_loss, cluster_loss] = get_values(log_file_path)
	try:
		suffix = folder.split('_')[3]
	except Exception as e:
		suffix = 'exp'
	x = range(len(nmi))
	print('plotting nmi')
	ax1.plot(x, nmi, label='nmi_'+suffix)
	print('plotting convloss')
	x = range(len(conv_loss))
	ax2.plot(x, conv_loss, label='convLoss_'+suffix)
	print('plotting clusterloss')
	x = range(len(cluster_loss))
	ax3.plot(x, cluster_loss, label='clusterLoss_'+suffix)

ax1.legend(loc='upper right')
ax2.legend(loc='top right')
ax3.legend(loc='top right')

plt.show()

fig1.savefig('nmi_plots.jpg')
fig2.savefig('convloss_plots.jpg')
fig3.savefig('clusterloss_plots.jpg')
