
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def load_SoN_images(SoN_dir, img_folder_names):
    im_paths = []
    son_avg = []
    son_var = []    
    SoN_imgs_path = os.path.join(SoN_dir, 'images')
    with open(os.path.join(SoN_dir,'votes.tsv'), 'r') as csvfile:
        SoN_reader = csv.reader(csvfile, delimiter='\t')
        next(SoN_reader) # Skip header
        for row in SoN_reader:
            for image_folder in img_folder_names:
                im_path = os.path.join(SoN_imgs_path, image_folder, row[0]+'.jpg')
                if os.path.isfile(im_path):
                    im_paths.append(im_path)
                    son_avg.append(np.float32(row[3]))
                    son_var.append(np.float32(row[4]))
    return im_paths, son_avg, son_var

def inspect_dataset(dataset,attr_names):
    fig, axs = plt.subplots(nrows=1, ncols=2)
    for idx in range(20, 2000, 20):
        print(idx)
        show_image(dataset.__getitem__(idx), attr_names, fig, axs)
        axs[0].cla()
        axs[1].cla()

def show_image(sample,names,fig,axs):
    axs[0].imshow(sample['image'])
    label = sample['label']
    names = [names[i] for i in np.where(label>0)[0]]
    label = label[label>0]
    axs[1].barh(np.arange(len(label)),label)
    axs[1].set_yticks(np.arange(len(label)))
    axs[1].set_yticklabels(names)
    fig.tight_layout()
    plt.pause(0.0001)
    plt.waitforbuttonpress(timeout=-1)