import pandas as pd
import matplotlib.pyplot as plt


def loss_plot(ld_filename, sv_filename):

    data = pd.read_csv('output/txtfiles/{}'.format(ld_filename))

    patch_loss = data.patch_loss
    index = range(0, len(patch_loss))

    plt.figure(num=1, figsize=(16, 6), dpi=100, facecolor='w', edgecolor='b')
    plt.scatter(index, patch_loss)
    plt.savefig('output/figures/{}'.format(sv_filename), format='eps')
    plt.show()
