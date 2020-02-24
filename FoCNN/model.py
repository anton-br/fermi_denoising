#!/usr/bin/env python
# coding: utf-8

import sys, os, os.path
import random
import pickle
import argparse
import astropy.wcs as wcs
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize

from sys import argv
from astropy.io import fits
from scipy.stats import poisson
from random import shuffle

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.autograd import Function, Variable
import torch
import torch.nn as nn


import warnings
warnings.filterwarnings('ignore')
torch.cuda.set_device(0)
device = torch.device('cuda')
torch.cuda.empty_cache()


_fname_prefold = ''
_fname_data = 'data/'
_fname_database = _fname_prefold + _fname_data
inter = 0

inter = 16
taille = 20

def show(image):
    plt.imshow(image)
    plt.show()   

def fft(img, x=9, x1=5):        
    # Applies Fast Fourier Tranform to img, cuts freuences lower than x1 and higher than x2, applied inverted Fast Fourier Transform.
    try:
        rows, cols = img.shape
    except:
        try:
            n, rows, cols = img.shape
        except:
            img = img.reshape(img.shape[0]*img.shape[1], img.shape[2], img.shape[3])
            n, rows, cols = img.shape
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    
    crow,ccol = int(rows/2) , int(cols/2)

    fshift[:, crow-x1:crow+x1, ccol-x1:ccol+x1] = 0
    
    new_shift = np.zeros((img.shape))
    new_shift[:, crow-x:crow+x, ccol-x:ccol+x] = fshift[:, crow-x:crow+x, ccol-x:ccol+x]

    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back



def divideMatriceList(A, w, intersection=0):
    s = np.max(A.shape)
    new_As = []
    color = np.max(A)
    for i in range(0, s-w+1, w-intersection):
        for j in range(0, s-w+1, w-intersection):
            A_small = np.ones((1, w, w))*(i*w + j)*color/w**2
            A_small = np.copy(A[0, i:i+w, j:j+w])
            new_As.append(np.copy(A_small))
    return new_As

def divideMatriceListY(A, w):
    s = np.max(A.shape)
    new_As = []
    for i in range(0, s, w):
        for j in range(0, s, w):
            A_small = np.zeros((1, w, w))
            A_small = A[0, i*w:i*w+w, j*w:j*w+w]
            new_As.append(A_small)
    return new_As



def divideMatrices(X, Y, w, test = 1, intersection = 0):
    #X and Y are images, X is a noised data and Y is a deconvolutional image.
    #Both are cut on wxw images with intersection.
    Xs = []
    Ys = []
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        pot_x = divideMatriceList(x, w, intersection)
        pot_y = divideMatriceList(y, w, intersection)
        for yi in range(len(pot_y)):
            if not test and ((np.sum(pot_y[yi]) == 0) and np.random.rand() > 0.1):
                continue
            Xs.append(pot_x[yi])
            Ys.append(pot_y[yi])
    return Xs, Ys


def toGray(x):
    s = np.max(x.shape)
    image = x.reshape(s, -1)
    return image

_losses = []
def addMask(x, y):
    masked_x = np.zeros((len(x), len(x[0])))
    K = 1
    S = int(K / 2)
    T = K - S
    for pair in y:
        pair = pair.astype(int)
        i,j = pair[0], pair[1]
        masked_x[np.max([0, i-S]):np.min([199,i+T]), np.max([0, j-S]):np.min([199,j+T])] = 1
    return masked_x
        
def to3D(x):
    s = np.max(x.shape)
    image = x.reshape((1, s,s))
    return image


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(        
            nn.Conv2d(
                in_channels=4,     
                out_channels=8,           
                kernel_size=3,           
                stride=1,               
                padding=1,                 
            ),  
            
            nn.ReLU(), 
            
        )
        
        self.conv2 = nn.Sequential(       
            nn.Conv2d(8, 32, 5, 1, 2),    
            nn.ReLU(),                    
        )
        
        self.conv3 = nn.Sequential(       
            nn.Conv2d(32, 63, 3, 1, 1),    
            nn.ReLU(),         
        )

        self.conv4 = nn.Sequential(       
            nn.Conv2d(64, 128, 3, 1, 2),    
            nn.ReLU(),                    
            nn.MaxPool2d((2,2)),
        )
        self.conv5 = nn.Sequential(       
            nn.Conv2d(128, 256, 3, 1, 2),    
            nn.ReLU(),                    
            nn.MaxPool2d((2,2)),
        )

        
        
        self.out = nn.Sequential( 
            nn.Linear(9216, 400),  
            nn.Sigmoid()
        )
        self.k = 0
        self.cuda()
        self.to(device)

    def forward(self, x,xs):
        start = x
        for x2 in xs:
            x = torch.cat((x, x2), dim=1)
        x = self.conv1(x) 
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.cat((x, start), dim=1)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.size(0), -1)          
        output = self.out(x)
        return output, x

def change(x0, y0, x1, y1, reduced,im_np):
    if x1 >= 200 or y1 >= 200 or x1 < 0 or y1 < 0:
        return reduced
    proba = im_np[x0, y0]
    if reduced[x1,y0]==1:
        if im_np[x1, y1] < proba:
            reduced[x1, y1] = 0
        else:
            reduced[x0, y0] = 0
    return reduced

def localize(x0, y0, reduced, im_np):
    reduced = change(x0, y0, x0-1, y0, reduced,im_np)
    reduced = change(x0, y0, x0+1, y0, reduced,im_np)
    reduced = change(x0, y0, x0, y0-1, reduced,im_np)
    reduced = change(x0, y0, x0, y0+1, reduced,im_np)
    return reduced
    
    
    
def keepPoints(reduced, im_np, x, y_):
    n = len(x)
    for i in range(n):
        for j in range(n):
            reduced = localize(x[i], y_[i], reduced, im_np)
            
    return reduced


     

class model:
    def __init__(self, shared=''):
        self.num_train_samples=0
        self.num_labels=1
        self.is_trained=False
        self.shared=shared
        self.cnn = CNN()
        self._losses = []
        self.EPOCH = 12000            
        self.BATCH_SIZE = 32
        self.LR = 0.001
        self.save_freq = 1
        self.optimizer = torch.optim.Adam(self.cnn.parameters(), lr=self.LR)  
        self.k=0
        #print(self.cnn)

    def preprocess_for_train(self, X, y):
        plotting = 0
        y_3D = [to3D(addMask(toGray(X[i]), y[i])) for i in range(len(X))]
       
        x_3D = np.array(X)
        y_3D = np.array(y_3D)

        w=20

        x_train, y_train = divideMatrices(x_3D, y_3D, w, 0, 5)


        x_train = np.array(x_train)
        y_train = np.array(y_train)
        if plotting:
            plt.imshow(y_train[0].reshape(20,-1))
            plt.show()          
        additional_x = []

        x_train_fft = fft(x_train)
        additional_x.append(x_train_fft)

        x_train_fft = fft(x_train, x=100, x1=7)
        additional_x.append(x_train_fft)

        x_train_fft = fft(x_train, x=100, x1=9)
        additional_x.append(x_train_fft)
        
        x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
        for i in range(len(additional_x)):
            additional_x[i] = additional_x[i].reshape(x_train.shape)
        additional_x = np.array(additional_x)
        y_train = y_train.reshape(y_train.shape[0], 1, y_train.shape[1], y_train.shape[2])

        new_Y_train = []
        for i in range(len(y_train)):

            data = y_train[i]

            size = 20
            data_list = [0]*size*size

            x,y, z = np.where(data >0)
            for index in range(len(x)):
                data_list[y[index]*size+z[index]] = 1.0

            data_fl = np.array(data_list)

            new_Y_train.append(data_fl)
        y_train = np.array(new_Y_train)


        new_x_train = []
        new_y_train = []
        new_x_train_fft = []
        for i in range(len(x_train)):
            c = x_train[i].astype(float)
            new_x_train.append(c*1.0/np.max(x_train[i]))
            for j in range(len(additional_x)):
                c = additional_x[j][i].astype(float)
                additional_x[j][i] = c*1.0/np.max(additional_x[j][i])          
            if np.max(y_train[i]) == 0:
                new_y_train.append(y_train[i])
                continue
            c = y_train[i].astype(float)
            new_y_train.append(c*1.0/np.max(y_train[i]))
        x_train = np.array(new_x_train)
        y_train = np.array(new_y_train)  
        if plotting:
            plt.imshow(new_y_train[0].reshape(20,-1))
            plt.show()    
        for i in range(len(additional_x)):
            additional_x[i] = np.array(additional_x[i])

        x_train = torch.Tensor(x_train)
        y_train = torch.Tensor(y_train)
        for i in range(len(additional_x)):
            additional_x[i] = torch.Tensor((additional_x[i]))
        return x_train, y_train, additional_x

    def fit(self, X, y):
        #X and y are lists of paths to images and corresponding coordinates
        EPOCH = self.EPOCH
        LR = self.LR
        batch_size = self.BATCH_SIZE     


        self.cnn.k = self.k
        cnn = self.cnn

        optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   
        loss_func = nn.modules.loss.MSELoss(2)
        base = self.cnn.k

        pairs = list(zip(X, y))
        shuffle(pairs)
        X, y = zip(*pairs)

        for epoch in range(base, base + EPOCH):
            print('Epoch: ', epoch)
            for step in tqdm.tqdm(range(int(len(X)/batch_size))): 
                x_batch = [np.load(x_name).reshape(1, 200, 200) for x_name in X[batch_size*step:batch_size*step + batch_size]]
                y_batch = [np.load(y_name) for y_name in y[batch_size*step:batch_size*step + batch_size]]
                b_x, b_y, additional_x = self.preprocess_for_train(x_batch, y_batch)  
                b_xs = []
                                
                b_x = Variable(b_x, requires_grad=True).cuda()
                b_y = b_y.cuda()
                for i in range(len(additional_x)):
                    b_xs.append(Variable(torch.Tensor(additional_x[i]), requires_grad=True).cuda())
                output, _ = cnn(b_x, b_xs)             

                self.loss = ( nn.modules.loss.L1Loss()(output, b_y)).cuda()
                self.optimizer.zero_grad()          
                self.loss.backward()
     
                self._losses.append(self.loss.data.cpu().numpy())

                self.optimizer.step()           

            if epoch % self.save_freq ==0 and epoch >= 0:
                self.cnn.k = epoch
                self.k = epoch
                self.save(k = epoch)


        self.is_trained=True
        print("Training is done!")
    def preprocess_for_test(self, X):
        x_test = []
        for i in range(len(X)):
            x_test.extend(divideMatriceList(X[i],20, inter))
        x_test = np.array(x_test)
        additional_x = []

        x_test_fft = fft(x_test)
        additional_x.append(x_test_fft)

        x_test_fft = fft(x_test, x=100, x1=7)
        additional_x.append(x_test_fft)

        x_test_fft = fft(x_test, x=100, x1=9)
        additional_x.append(x_test_fft)

        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[2], x_test.shape[2])

        for i in range(len(additional_x)):
            additional_x[i] = additional_x[i].reshape(x_test.shape)

        new_x_test = []
        new_x_test_fft = []
        for i in range(len(x_test)):
            c = x_test[i].astype(float)
            new_x_test.append(c*1.0/np.max(x_test[i]))

            for j in range(len(additional_x)):
                c = additional_x[j][i].astype(float)
                additional_x[j][i] = c*1.0/np.max(additional_x[j][i])    

        x_test = np.array(new_x_test)
        x_test = torch.Tensor(x_test).cuda()

        for i in range(len(additional_x)):
            additional_x[i] = np.array(additional_x[i])

        for i in range(len(additional_x)):
            additional_x[i] = torch.Tensor((additional_x[i])).cuda()
        return x_test, additional_x


    def predict(self, X, threshold = 0.8):
        x_test, additional_x = self.preprocess_for_test(X)
        y = (self.cnn(x_test, additional_x))[0].data.cpu().numpy()
 
        line_n  = int((200-taille)/(taille - inter)+1) 
        answers = []
        images = []
        for j in range(0, len(y), line_n**2):
            im_l = []
            im_np = np.zeros((200,200))
            multi_np = np.zeros((200,200))
            for i in range(j, j+line_n**2, line_n):
                
                img = y[i:i+line_n]
                final_img = np.zeros((taille,200))
                multi_img = np.zeros((taille,200))
                if img.size == taille*taille*line_n:
                    im = (img.reshape((line_n, taille,taille)))

                for k in range(line_n):
                    final_img[:, k*(taille-inter):k*(taille-inter)+taille] += im[k]
                    multi_img[:, k*(taille-inter):k*(taille-inter)+taille] += 1
                im_np[(i//line_n)*(taille-inter):(i//line_n)*(taille-inter)+taille, :] += final_img
                multi_np[(i//line_n)*(taille-inter):(i//line_n)*(taille-inter)+taille, :] += multi_img
            im_np = (im_np/(multi_np))*1.0

            threshold = 0.1
            vplot= 0
            if vplot:
                plt.imshow(im_np)
                plt.show()

                plt.imshow(X[j//100].reshape(200,-1))
                plt.show()
                plt.imshow(im_np>threshold)
                plt.show()

            first_thres = 0.6
            im_np[:2,:]=(im_np[:2,:] > first_thres)
            im_np[:,:2]=(im_np[:,:2] > first_thres)
            im_np[200-2:,:]=(im_np[200-2:,:] > first_thres)
            im_np[:,200-2:]=(im_np[:,200-2:] > first_thres)  


            im_np[:5,:]=(im_np[:5,:] > 0.11)
            im_np[:,:5]=(im_np[:,:5] > 0.11)
            im_np[200-5:,:]=(im_np[200-5:,:] > 0.11)
            im_np[:,200-5:]=(im_np[:,200-5:] > 0.11)  
          
            x,y_ = np.where(im_np > threshold)
            reduced = (im_np>threshold)
            reduced = keepPoints(reduced, im_np, x, y_)
            if vplot:
                plt.imshow(im_np)
                plt.show()
            x,y_ = np.where(reduced > threshold)

            answer = np.array([np.array([x[k], y_[k]]) for k in range(len(x))])
            answers.append(answer)
            images.append(im_np)
        return answers
    
    def save(self, path="./weights",k='0'):
        if not os.path.exists(path):
            os.mkdir(path)
        print('saving')
        self.k = self.cnn.k
        pickle.dump(self, open(os.path.join(path, 'class_model.pickle'), "wb"), 2) 
        torch.save({
                  'state_dict': self.cnn.state_dict(),
                  'loss': self.loss,
                  'optimizer' : self.optimizer.state_dict(),
               }, os.path.join(path, 'torch_model_all'+ str(self.cnn.k)+ '.pt'))

        print("Model saved to: " + os.path.join(path, 'torch_model_all'+ str(self.cnn.k)+ '.pt'))

    def load(self, path="./weights/", k=-1, weights=None):
        modelfile = path + 'class_model.pickle'
        if os.path.isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            if k != -1:
                self.k = k

        if weights is None:
            PATH = (path + 'torch_model_all' + str(self.k) + '.pt')       
        else:
            PATH = weights      
        checkpoint = torch.load(PATH)

        self.cnn.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        self.k += 1
        self.cnn.k = self.k
        self.loss = checkpoint['loss'].cuda()

        return self




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Score results with known reference (for training).')
    parser.add_argument('--mode', nargs='?', default='train', choices=('train', 'test'), type=str, help='training or testing mode')
    parser.add_argument('--w', nargs='?', default='None', type=str, help='weights for the neural network')
    parser.add_argument('--data_path', nargs='?', const='../data', default='../data', type=str, help='path to the data folder with both input and reference folders')
    parser.add_argument('--i', nargs='?', const='input_data', default='input_data', type=str, help='an input folder with inputs')
    parser.add_argument('--r', nargs='?', const='reference_data', default='reference_data', help='a reference folder with outputs for training')
    parser.add_argument('--o', nargs='?', const='output_data', default='output_data', type=str, help='a folder where predictions will be saved')
    parser.add_argument('--s', nargs='?', const='test', default='test', type=str, help='subfolder of the data set which should be evaluated (both folders given by --i and --r should contain a folder with this name')
    args = parser.parse_args(argv[1:])

    x_path = os.path.join(args.data_path, args.i, args.s)
    y_path = os.path.join(args.data_path, args.r, args.s)

    names = os.listdir(x_path) 

    x_names = [os.path.join(x_path, name) for name in os.listdir(x_path)]  
    y_names = [os.path.join(y_path, name) for name in os.listdir(y_path)]    

    import tqdm
    X = []
    Y = []
    model0 = model()
    if args.w != 'None':
        model0 = model0.load(weights=args.w)
    
        

    for name in tqdm.tqdm(names):
        if name.find('jpg') != -1 or name.find('T') != -1:

            continue
        try:
            x = np.load(os.path.join(x_path, name))
            x = x.reshape(1, 200, 200)
            y = np.load(os.path.join(y_path, name))
            #print(name)
        except:
            continue
        
        X.append(x)
    
        
        Y.append(y)
    print(len(X))

    if (args.mode == 'train'):
        model0.fit(x_names, y_names)
    if (args.mode == 'test'):
        if not os.path.exists(args.o):
            os.mkdir(args.o)
        if not os.path.exists(os.path.join(args.o, args.s)):
            os.mkdir(os.path.join(args.o, args.s))
        for x_name in os.listdir(x_path):
            x = np.load(os.path.join(x_path, x_name)).reshape(1, 1, 200, 200)
            y_pred = model0.predict(x)[0]
            np.save(os.path.join(args.o, args.s, x_name), y_pred)
           
    




