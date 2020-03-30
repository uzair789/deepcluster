import matplotlib.pyplot as plt

root = '/home/biometrics/deepcluster-git/deepcluster/'


folders = [root+"""exp_uz_resnet_K1000_seqFix/linear_classif_layer3_50176feat/log_conv_layer3_50176feat.txt""",
           root+"""exp_uz_resnet_K1000_seqFix/linear_classif_layer2_feat/log_conv_layer2_feat.txt""",
           root+"""results/exp/linear_classif/log_conv5.txt"""]


def get_values(log_file_path):

    f = open(log_file_path, 'r')
    training_prec1 = []
    avg_training_prec1 = []
    val_prec1 = []
    avg_val_prec1 = []
    for line in f:
        line = line.rstrip()
        if 'Epoch:' in line:
            a = line.split('Prec@1')[1]
            temp = a.split(' ')
            #print(temp)
            b = temp[1]
            c = temp[2].split('\t')[0][1:-1]
            training_prec1.append(float(b))
            avg_training_prec1.append(float(c))
        if 'Validation:' in line:
            a = line.split('Prec@1')[1]
            temp = a.split(' ')
            b = temp[1]
            c = temp[2].split('\t')[0][1:-1]

            #print('val', b, type(b))
            val_prec1.append(float(b))
            avg_val_prec1.append(float(c))

    return training_prec1, val_prec1, avg_training_prec1, avg_val_prec1


fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()

for folder in folders:
    # log_file_path = os.path.join(folder, 'log_conv5.txt')
    # f folder is actually the path to the log file
    [training_prec1, val_prec1, avg_train, avg_val] = get_values(folder)
    try:
        suffix = folder.split('/')
        suffix = suffix[-1].split('.txt')[0].split('_')[-1]
    except Exception as e:
        print(e)
        suffix = 'exp'
    x = range(len(training_prec1))
    print('plotting training prec1')
    ax1.plot(x, training_prec1, label='training_prec1_'+suffix)
    print('plotting val prec1')
    x = range(len(val_prec1))
    ax2.plot(x, val_prec1, label='val_prec1_'+suffix)

    x = range(len(avg_train))
    ax3.plot(x, avg_train, label='avg_train_prec1'+suffix)

    x = range(len(avg_val))
    ax4.plot(x, avg_val, label='avg_val_prec1_'+suffix)


ax1.legend(loc='upper right')
ax2.legend(loc='top right')
ax3.legend(loc='top right')
ax4.legend(loc='top right')

plt.show()

fig1.savefig('training_prec1_plots.jpg')
fig2.savefig('val_prec1_plots.jpg')
fig3.savefig('avg_train_prec1_plots.jpg')
fig4.savefig('avg_val_prec1_plots.jpg')
