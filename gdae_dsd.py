# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import cPickle as pickle
import sys
from ASP.IOMethods import AudioIO as IO
from ASP import TFMethods as TF
from ASP.MaskingMethods import FrequencyMasking as fm

### Keras imports
from keras import backend as K
from keras.layers import Input, Highway, Lambda, merge,\
    GRU, Lambda, TimeDistributed, Dense
from keras.callbacks import ModelCheckpoint as MC
from keras.models import Model, Sequential
from keras.regularizers import activity_l1, activity_l2, l1, l2
from keras import optimizers as opt
opt = opt.adam(clipnorm = 0.35)             # Optimizer

def prepare_olapsequences(ms, vs, lsize, olap, bsize):
    from numpy.lib import stride_tricks
    global trimframe
    trimframe = ms.shape[0] % (lsize - olap)
    print(trimframe)
    if trimframe != 0:
        ms = np.pad(ms, ((0,trimframe), (0,0)), 'constant', constant_values=(0,0))
        vs = np.pad(vs, ((0,trimframe), (0,0)), 'constant', constant_values=(0,0))

    ms = stride_tricks.as_strided(ms, shape=(ms.shape[0] / (lsize - olap), lsize, ms.shape[1]),
                             strides=(ms.strides[0] * (lsize - olap), ms.strides[0], ms.strides[1]))
    ms = ms[:-1, :, :]

    vs = stride_tricks.as_strided(vs, shape=(vs.shape[0] / (lsize - olap), lsize, vs.shape[1]),
                                  strides=(vs.strides[0] * (lsize - olap), vs.strides[0], vs.strides[1]))
    vs = vs[:-1, :, :]

    btrimframe = (ms.shape[0] % bsize)
    if btrimframe != 0:
        ms = ms[:-btrimframe, :, :]
        vs = vs[:-btrimframe, :, :]

    #print(ms.max(), ms.min(), vs.max(), vs.min())
    #print(ms.shape, vs.shape)
    return ms, vs

def build_GRU(dim, lsize, bsize, cost = 'mse'):
    def filt(args):
        return subsample(args[0]) * K.abs(subsample(args[1]))

    def subtract(args):
        return K.abs(args[0] - args[1])

    def myfun(args):
        return K.abs(subsample(args[0]))

    def rect(args):
        return K.abs(args[0])

    def flipsum(args):
        return args[0][:, ::-1, :] + args[1]

    def subsample(args):
        return args[:, 3:15, :]

    xin = Input(batch_shape = (bsize, lsize, dim))

    # Encoding Part using bi-directional GRUs
    rnnAF = GRU(dim, init='glorot_normal', activation = 'tanh',
        inner_activation='hard_sigmoid', W_regularizer=None,
               U_regularizer=None, stateful = False, return_sequences = True, consume_less = 'gpu') (xin)
    mr = merge([xin, rnnAF], mode = 'sum')

    rnnBF = GRU(dim, init='glorot_normal', activation = 'tanh',
        inner_activation='hard_sigmoid', W_regularizer=None,
               U_regularizer=None, go_backwards = True, return_sequences = True, consume_less = 'mem') (xin)
    mrB = merge([xin, rnnBF], mode = flipsum, output_shape = (lsize, dim))

    # The return of Bi-GRUs
    mrBDIR = merge([mr, mrB], mode = 'concat')

    # Decoding part
    rnnDsv = GRU(dim, init='glorot_normal', activation = 'tanh',
        inner_activation='hard_sigmoid', W_regularizer=None, U_regularizer=None,
                 stateful = False, return_sequences = True, consume_less = 'gpu') (mrBDIR)
    svout = merge([xin, rnnDsv], output_shape = (lsize/2, dim), mode = filt)

    # Post filtering with sparsity constraint
    hC = TimeDistributed(Highway(input_dim = dim, activation='relu', activity_regularizer=activity_l2(1e-4))) (svout)

    model = Model(xin, [svout, hC, rnnDsv])

    # Cost Functions
    def mseloss(ytrue, ypred):
        # Update of the true
        ytrue = xin * (K.pow(ytrue, 1.) + K.epsilon())/(K.pow(xin, 1.) + K.epsilon())

        return K.sum(K.pow(ytrue-ypred,2.), axis = -1)

    def KL(ytrue, ypred):
        # Update of the true
        ytrue = subsample(xin) * (K.pow(ytrue, 1.) + K.epsilon())/(K.pow(subsample(xin), 1.) + K.epsilon()) + 1e-6
        ypred += 1e-6

        return K.sum(ytrue * (K.log(ytrue) - K.log(ypred)) + (ypred - ytrue), axis=-1)


    def KLbkg(ytrue, ypred):
        # Update of the true
        ytrue = subsample(xin) * K.abs(1. - ((K.pow(ytrue, 1.) + K.epsilon())/(K.pow(subsample(xin), 1.) + K.epsilon()))) + 1e-6
        ypred += 1e-6

        return K.sum(ytrue * (K.log(ytrue) - K.log(ypred)) + (ypred - ytrue), axis=-1)

    def IS(ytrue, ypred):
        # Update of the true
        ytrue = xin * (K.pow(ytrue, 2.) + K.epsilon())/(K.pow(xin, 2.) + K.epsilon()) + 1e-6
        ypred += 1e-6

        return K.mean((ytrue/ypred) - (K.log(ytrue) - K.log(ypred)) - 1.,axis=-1)

    if cost == 'mse':
        print('MSE')
        model.compile(optimizer = opt, loss = [mseloss, mseloss])
    elif cost == 'kl':
        print('Kullback-Leibler')
        model.compile(optimizer = opt, loss = [KL, KL])
    elif cost == 'klbkg':
        print('Kullback-Leibler for Accompaniment Instrument')
        model.compile(optimizer = opt, loss = [KLbkg, KLbkg])
    elif cost == 'is':
        print('Itakura-Saito')
        model.compile(optimizer = opt, loss = IS)

    return model

def build_bkgGRU(dim, lsize, bsize, cost = 'mse'):
    xin = Input(batch_shape = (bsize, lsize, dim))

    # Encoding Part using bi-directional GRUs
    rnnAF = GRU(dim, init='glorot_normal', activation = 'tanh',
        inner_activation='hard_sigmoid', W_regularizer=None,
               U_regularizer=None, stateful = False, return_sequences = True, consume_less = 'gpu') (xin)
    mr = merge([xin, rnnAF], mode = 'sum')

    rnnBF = GRU(dim, init='glorot_normal', activation = 'tanh',
        inner_activation='hard_sigmoid', W_regularizer=None,
               U_regularizer=None, go_backwards = True, return_sequences = True, consume_less = 'mem') (xin)
    mrB = merge([xin, rnnBF], mode = flipsum, output_shape = (lsize, dim))

    # The return of Bi-GRUs
    mrBDIR = merge([mr, mrB], mode = 'concat')

    # Decoding part
    rnnDsv = GRU(dim, init='glorot_normal', activation = 'tanh',
        inner_activation='hard_sigmoid', W_regularizer=None, U_regularizer=None,
                 stateful = False, return_sequences = True, consume_less = 'gpu') (mrBDIR)
    svout = merge([xin, rnnDsv], output_shape = (lsize/2, dim), mode = filt)

    # Post filtering with sparsity constraint
    hC = TimeDistributed(Highway(input_dim = dim, activation='relu', activity_regularizer=None)) (svout)

    model = Model(xin, [svout, hC])

    # Cost Functions
    def mseloss(ytrue, ypred):
        # Update of the true
        ytrue = xin * (K.pow(ytrue, 1.) + K.epsilon())/(K.pow(xin, 1.) + K.epsilon())

        return K.sum(K.pow(ytrue-ypred,2.), axis = -1)

    def KL(ytrue, ypred):
        # Update of the true
        ytrue = subsample(xin) * (K.pow(ytrue, 1.) + K.epsilon())/(K.pow(subsample(xin), 1.) + K.epsilon()) + 1e-6
        ypred += 1e-6

        return K.sum(ytrue * (K.log(ytrue) - K.log(ypred)) + (ypred - ytrue), axis=-1)


    def KLbkg(ytrue, ypred):
        # Update of the true
        ytrue = subsample(xin) * K.abs(1. - ((K.pow(ytrue, 1.) + K.epsilon())/(K.pow(subsample(xin), 1.) + K.epsilon()))) + 1e-6
        ypred += 1e-6

        return K.sum(ytrue * (K.log(ytrue) - K.log(ypred)) + (ypred - ytrue), axis=-1)

    def IS(ytrue, ypred):
        # Update of the true
        ytrue = xin * (K.pow(ytrue, 2.) + K.epsilon())/(K.pow(xin, 2.) + K.epsilon()) + 1e-5
        ypred += 1e-5

        return K.mean((ytrue/ypred) - (K.log(ytrue) - K.log(ypred)) - 1.,axis=-1)

    if cost == 'mse':
        print('MSE')
        model.compile(optimizer = opt, loss = [mseloss, mseloss])
    elif cost == 'kl':
        print('Kullback-Leibler')
        model.compile(optimizer = opt, loss = [KL, KL])
    elif cost == 'klbkg':
        print('Kullback-Leibler for Accompaniment Instrument')
        model.compile(optimizer = opt, loss = [KLbkg, KLbkg])
    elif cost == 'is':
        print('Itakura-Saito')
        model.compile(optimizer = opt, loss = IS)

    return model

if __name__ == '__main__':
    np.random.seed(218)
    analyseData = False
    N = 2048
    hop = 256
    nfiles = 50
    fs = 44100
    cfr = 1
    seqlen = 18
    bsize = 16
    cost = str(sys.argv[1]) #cfunctions = ['mse', 'kl', 'is'] # Avoid iterations because of GPU memory issues.
    epochs = 65

    if analyseData :
        MixturesPath = '/home/avdata/audio/own/dsd100/DSD100/Mixtures/'
        SourcesPath = '/home/avdata/audio/own/dsd100/DSD100/Sources/'
        foldersList = ['Dev', 'Test']
        # Usage of full dataset
        keywords = ['bass.wav', 'drums.wav', 'other.wav', 'vocals.wav', 'mixture.wav']
        # Usage of segmented dataset
        #keywords = ['bass_seg.wav', 'drums_seg.wav', 'other_seg.wav', 'vocals_seg.wav', 'mixture_seg.wav']
        foldersList = ['Dev', 'Test']
        # Generate full paths for dev and test
        DevMixturesList = sorted(os.listdir(MixturesPath + foldersList[0]))
        DevMixturesList = [MixturesPath + foldersList[0] + '/' + i for i in DevMixturesList]
        DevSourcesList = sorted(os.listdir(SourcesPath + foldersList[0]))
        DevSourcesList = [SourcesPath + foldersList[0] + '/' + i for i in DevSourcesList]

        TestMixturesList = sorted(os.listdir(MixturesPath + foldersList[1]))
        TestMixturesList = [MixturesPath + foldersList[1] + '/' + i for i in TestMixturesList]
        TestSourcesList = sorted(os.listdir(SourcesPath + foldersList[1]))
        TestSourcesList = [SourcesPath + foldersList[1] + '/' + i for i in TestSourcesList]

        # Extend Lists for full validation
        DevMixturesList.extend(TestMixturesList)
        DevSourcesList.extend(TestSourcesList)

        # Masking Threhsold
        pm = TF.PsychoacousticModel(N, nfilts = 32)
        for indx in tqdm(xrange(len(DevMixturesList[:nfiles]))):
            vox, _ = IO.wavRead(os.path.join(DevSourcesList[indx], keywords[3]), mono=True)
            bass, _ = IO.wavRead(os.path.join(DevSourcesList[indx], keywords[0]), mono=True)
            drms, _ = IO.wavRead(os.path.join(DevSourcesList[indx], keywords[1]), mono=True)
            oth, _ = IO.wavRead(os.path.join(DevSourcesList[indx], keywords[2]), mono=True)

            mix = vox + bass + drms + oth
            bkg = bass + drms + oth

            msseg, _ = TF.TimeFrequencyDecomposition.STFT(mix, sig.hamming(1025, True), N, hop)
            vsseg, _ = TF.TimeFrequencyDecomposition.STFT(vox, sig.hamming(1025, True), N, hop)
            bkseg, _ = TF.TimeFrequencyDecomposition.STFT(bkg, sig.hamming(1025, True), N, hop)

            if indx == 0 :
                vstrain = vsseg[3:-3, :]
                mstrain = msseg[3:-3, :]
                bktrain = bkseg[3:-3, :]

            else :
                mstrain = np.vstack((mstrain, msseg[3:-3]))
                vstrain = np.vstack((vstrain, vsseg[3:-3]))
                bktrain = np.vstack((bktrain, bkseg[3:-3]))

            del mix, vox, bass, drms, oth, msseg, vsseg
    else :
        print('Loading Data')
        try :
            mstrain = np.load('/mnt/IDMT-WORKSPACE/DATA-STORE/mis/Datasets/DSD-PQMF/mstrain.npy')
            vstrain = np.load('/mnt/IDMT-WORKSPACE/DATA-STORE/mis/Datasets/DSD-PQMF/vstrain.npy')
        except IOError:
            print('Loading from avdata')
            mstrain = np.load('/home/avdata/audio/own/dsd100/Analysed/Stereo/STFT/mstrain.npy')
            vstrain = np.load('/home/avdata/audio/own/dsd100/Analysed/Stereo/STFT/vstrain.npy')


    # Preparing sequences with overlap
    mstrain, vstrain = prepare_olapsequences(mstrain, vstrain, seqlen, 6, bsize)
    vstrain = vstrain[:, 3:15, :]
    dim = mstrain.shape[2]
    print(dim)

    ### DAE based on Residual GRU connections
    print('Constructing GRU')
    G = build_GRU(dim, seqlen, bsize, cost)
    G.fit(mstrain, [vstrain, vstrain], nb_epoch = epochs, batch_size = bsize, shuffle = False,
      callbacks=[MC('Gsv_bigru_'+cost+str(seqlen)+'.hdf5', monitor = 'loss', mode = 'min', period = 2, save_best_only = True)])