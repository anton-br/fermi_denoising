import os
import numpy as np
import matplotlib.pyplot as plt

def maskImage(x):
        x_test = np.zeros((200,200))
        for i in range(len(x)):
            x[i] = [x[i][0], x[i][1] ]
            l = 1.0
            c = x[i]
            x_test[(c[0])-int(l//2):(c[0])+int((l+1)//2), (c[1])-int(l//2):(c[1])+int((l+1)//2)] = 1
        return x_test 

def drawImages(name_im1, name_ref1, name_im2, name_ref2, cmap='cubehelix'):
    
    im1 = np.load(name_im1)
    ref1 = np.load(name_ref1)
    im2 = np.load(name_im2)
    ref2 = np.load(name_ref2)
    
    mask1 = maskImage(ref1)
    mask2 = maskImage(ref2)

    titles = ['Image#1', 'Image#2','Mask#1',  'Mask#2']
    images = [im1, im2, mask1, mask2]
    refs = [ref1, ref2]

    if True:
        i = 0
        fig, axs = plt.subplots(figsize=(20,20),nrows=2, ncols=2, constrained_layout=False)
        for ax in axs.flatten():
            ax.set_title(titles[i], fontsize=18)
            if (i >= 2):
                ax.imshow(images[i-2], cmap)
                for c in refs[i-2]:
                    ax.plot(c[1], c[0], marker='x', color='white', markersize=20)
            else:
                ax.imshow(images[i], cmap)
            
            i = i + 1
    #plt.savefig('example_dataset.png')
    plt.show()


