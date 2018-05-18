import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

original = np.load('appendix/MNIST non-targeted 40k.npz')
print(original.keys())
Original_img_o, Adv_o,Target_labels_o = (original['X_test'], original['X_adv'], original['adv_labels'])
protected = np.load('appendix/Defended MNIST non-targeted 40k.npz')
Original_img_p, Adv_p, Target_labels_p = (protected['X_test'], protected['X_adv'], protected['adv_labels'])
print(Adv_o.shape, Adv_p.shape)

fig = plt.figure(figsize=(10, 2))
gs = gridspec.GridSpec(4, 10)
gs.update(wspace=0.25, hspace=0.)

for i in range(0,10):
    ax = fig.add_subplot(gs[0, i])
    m = Original_img_o[i].reshape((28,28))
    ax.imshow(m, interpolation='none', cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_xlabel('{0}'.format(Target_labels_o[i]), fontsize=16)

for i in range(0,10):
    ax = fig.add_subplot(gs[1, i])
    m = Adv_o[i].reshape((28,28))
    ax.imshow(m, interpolation='none',cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('{0}'.format(Target_labels_o[i]), fontsize=16)

for i in range(0,10):
    ax = fig.add_subplot(gs[2, i])
    m = Original_img_p[i].reshape((28,28))
    ax.imshow(m, interpolation='none', cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_xlabel('7', fontsize=16)

for i in range(0,10):
    ax = fig.add_subplot(gs[3, i])
    m = Adv_p[i].reshape((28,28))
    ax.imshow(m, interpolation='none', cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('{0}'.format(Target_labels_p[i]), fontsize=16)

#gs.tight_layout(fig)
plt.show()
