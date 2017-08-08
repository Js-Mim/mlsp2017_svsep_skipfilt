# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

import gdae_dsd as gdae
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from ASP import TFMethods as TF
from ASP import IOMethods as IO
import os, sys
import cPickle as pickle
from ASP.MaskingMethods import FrequencyMasking as fm
from mir_eval import separation as bssEval

def crawlDSD(fileNum, mono = True):
    """ A not so elegant function to acquire the mixture and the
    true targeted sources from the DSD100 dataset, from both
    training and evaluation.
    Args:
        filenum  : 	 (int) Spans from 1 to 100
        mono     :   (bool) Monaural summation, True or False
    Returns:
        mix      :   (1D ndarray) Time-domain waveform of the monaural mixture
        vox      :   (1D ndarray) Time-domain waveform of the monaural singing voice
        bkg      :   (1D ndarray) Time-domain waveform of the monaural background music
        fs       :   (int)        Sampling frequency
    """
    # 0 indexing
    fileNum -= 1
    # F-IDMT
    MixturesPath = '/home/avdata/audio/own/dsd100/DSD100/Mixtures/'
    SourcesPath = '/home/avdata/audio/own/dsd100/DSD100/Sources/'
    foldersList = ['Dev', 'Test']
    # Usage of full dataset
    keywords = ['bass.wav', 'drums.wav', 'other.wav', 'vocals.wav', 'mixture.wav']
    # Usage of segmented dataset
    #keywords = ['bass_seg.wav', 'drums_seg.wav', 'other_seg.wav', 'vocals_seg.wav', 'mixture_seg.wav']

    # Generate full paths for dev and test
    DevMixturesList = sorted(os.listdir(MixturesPath + foldersList[0]))
    DevMixturesList = [MixturesPath + foldersList[0] + '/' + i for i in DevMixturesList]
    DevSourcesList = sorted(os.listdir(SourcesPath + foldersList[0]))
    DevSourcesList = [SourcesPath + foldersList[0] + '/' + i for i in DevSourcesList]

    TestMixturesList = sorted(os.listdir(MixturesPath + foldersList[1]))
    TestMixturesList = [MixturesPath + foldersList[1] + '/' + i for i in TestMixturesList]
    TestSourcesList = sorted(os.listdir(SourcesPath + foldersList[1]))
    TestSourcesList = [SourcesPath + foldersList[1] + '/' + i for i in TestSourcesList]

    # Extend Lists (Testing sub-set will not be used for training. Training sub-set will not be used for evaluation.)
    DevMixturesList.extend(TestMixturesList)
    DevSourcesList.extend(TestSourcesList)

    print(DevMixturesList[fileNum])
    bass, fs = IO.AudioIO.wavRead(os.path.join(DevSourcesList[fileNum], keywords[0]), mono = mono)
    drm, _ = IO.AudioIO.wavRead(os.path.join(DevSourcesList[fileNum], keywords[1]), mono = mono)
    oth, _ = IO.AudioIO.wavRead(os.path.join(DevSourcesList[fileNum], keywords[2]), mono = mono)
    vox, _ = IO.AudioIO.wavRead(os.path.join(DevSourcesList[fileNum], keywords[3]), mono = mono)
    mix, fs = IO.AudioIO.wavRead(os.path.join(DevMixturesList[fileNum], keywords[4]), mono = mono)

    bkg = (bass + oth + drm)

    return mix, vox, bkg, fs

def _IS(X, Xhat):
    """ Itakura-Saito divergence between two magnitude spectra.
    Args:
        X   : 	(2D ndarray) True magnitude spectrum
        Xhat:   (2D ndarray) Estimated magnitude spectrum
    Returns:
        d   :   (float) Average distance between spectra
    """
    eps = np.finfo(np.float32).tiny
    r1 = (np.abs(X) + eps) / (np.abs(Xhat) + eps)
    lg = np.log((np.abs(X) + 1e-6)) - np.log((np.abs(Xhat) + 1e-6))
    return np.mean(r1 - lg - 1.)

def reshape_data(mX, pX):
    """
        An uggly but helpful function to make sure
        that the shapes are preserved and correct
        for masking.
    """
    mXL, pXL = gdae.prepare_olapsequences(mX[0, :, :].T, pX[0, :, :].T, seqlen, overlap, 1)
    mXR, pXR = gdae.prepare_olapsequences(mX[1, :, :].T, pX[1, :, :].T, seqlen, overlap, 1)

    mXL = mXL[:, 3:15, :]
    mXL = np.reshape(mXL, (mXL.shape[0]*12, mXL.shape[2]))
    mXR = mXR[:, 3:15, :]
    mXR = np.reshape(mXR, (mXR.shape[0]*12, mXR.shape[2]))
    pXL = pXL[:, 3:15, :]
    pXL = np.reshape(pXL, (pXL.shape[0]*12, pXL.shape[2]))
    pXR = pXR[:, 3:15, :]
    pXR = np.reshape(pXR, (pXR.shape[0]*12, pXR.shape[2]))

    mXout = np.zeros((2, mXL.shape[1], mXL.shape[0]), dtype = np.float32)
    pXout = np.zeros((2, pXL.shape[1], pXL.shape[0]), dtype = np.float32)
    mXout[0, :, :] = mXL.T
    mXout[1, :, :] = mXR.T
    pXout[0, :, :] = pXL.T
    pXout[1, :, :] = pXR.T

    return mXout, pXout

def estimate_sources(mX, pX, hop, maskingMode=1, synthesis = True):
    """ Estimate singing voice and background music using a trained
    supervised method based on GRU-denoising auto-encoding.
    Args:
        mX          : 	(2D ndarray) Mixture magnitude spectrum
        pX          :   (2D ndarray) Mixture phase spectrum
        hop         :   (int)        Hop size for STFT analysis & synthesis
        maskingMode :   (int)        0: Ideal Binary Mask will be used to recover the sources
                                     1: Soft time-frequency masking, by employing additive
                                     fractional power spectrograms of outcomes of two deep learning models,
                                     will be used to recover the sources.
                                     2: Soft time-frequency masking, by employing minimum correlation
                                     fractional power spectrograms of outcomes of two deep learning models,
                                     will be used to recover the sources
                                     3: Wiener filtering, by employing outcomes of two deep learning models
                                     4: Soft time-frequency masking, by employing minimum correlation
                                     fractional power spectrograms of outcomes of a single deep learning model,
                                     will be used to recover the sources
                                     5: No Filtering

        synthesis   :   (bool)       Time-domain synthesis
    Returns:
        svhat       :   (ndarray)    Etimated singing voice source
        bkhat       :   (ndarray)    Estimated background/accompaniment source
                                     (Will return a time-domain signal if synthesis == True,
                                      spectral representation otherwise)
    """
    # Data preparation
    mX, pX = gdae.prepare_olapsequences(mX, pX, seqlen, overlap, 1)
    # Paths
    bkgModel = 'trainedModels/Gsv_drbigru_klbkg18_ep40.hdf5'
    svModel = 'trainedModels/Gsv_drbigru_kl18_ep46.hdf5'

    print('Loading Solutions')
    GB = gdae.build_GRU(dim, seqlen, 1, cost = 'IS')
    GS = gdae.build_GRU(dim, seqlen, 1, cost = 'IS')
    GB.load_weights(bkgModel)
    GS.load_weights(svModel)

    print('Predicting')
    bkhat = GB.predict(mX)
    HWAoutB = np.abs(bkhat[1])
    HWAoutB.shape = (HWAoutB.shape[0]*12, dim)
    mask = np.abs(bkhat[2][:, 3:-3, :])
    mask.shape = (mask.shape[0]*12, dim)
    del bkhat
    svhat = GS.predict(mX)
    HWAoutS = np.abs(svhat[1])
    HWAoutS.shape = (HWAoutS.shape[0]*12, dim)
    masksv = np.abs(svhat[2][:, 3:-3, :])
    masksv.shape = (masksv.shape[0]*12, dim)
    del svhat

    # Reshaping
    mX = mX[:, 3:15, :]
    mX.shape = (mX.shape[0]*12, dim)


    pX = pX[:, 3:15, :]
    pX.shape = (pX.shape[0]*12, dim)

    if maskingMode == 0:
        print('Time Frequency Masking: GRU-DBM')
        mask = fm(mX, HWAoutS, HWAoutB, [], [], alpha = 1., method='IBM')

    elif maskingMode == 1:
        print('Pursuing the additivity property: GRU-DADM')
        calpha = np.arange(0.5, 2.1, step = 0.1)
        tempIS = []
        for indx in xrange(len(calpha)):
            Xhat = (HWAoutS ** calpha[indx]) + (HWAoutB ** calpha[indx])
            tempIS.append(_IS(mX**calpha[indx], Xhat))

        calpha = calpha[np.argmin(tempIS)]

        print('Time Frequency Masking')
        mask = fm(mX, HWAoutS, [HWAoutB], [], [], alpha = calpha, method='alphaWiener')

    elif maskingMode == 2:
        print('Pursuing minimum correlation')
        calpha = np.arange(0.9, 1.9, step = 0.1)
        tempIS = []
        for indx in xrange(len(calpha)):
            Xhat = (HWAoutS ** calpha[indx]) * (HWAoutB ** calpha[indx])
            tempIS.append(np.sum(Xhat ** (1./calpha[indx])))

        calpha = calpha[np.argmin(tempIS)]          # calpha in all studied cases is equal to 1.7

        print('Time Frequency Masking: GRU-D')
        mask = fm(mX, HWAoutS, [HWAoutB], [], [], alpha = calpha, method='alphaWiener')

    elif maskingMode == 3:
        print('Time Frequency Masking: GRU-DWF')
        mask = fm(mX, HWAoutS, [HWAoutB], [], [], alpha = 2., method='alphaWiener')

    elif maskingMode == 4:
        print('Time Frequency Masking: GRU-S')
        svhat = mX * ((HWAoutS ** 1.7 + 1e-16)/(mX ** 1.7 + 1e-16))
        bkhat = mX * (1. - ((HWAoutS ** 1.7 + 1e-16)/(mX ** 1.7 + 1e-16)))

    else:
        print('No Masking: Raw outputs')
        svhat = HWAoutS
        bkhat = HWAoutB

    try:
        svhat = mask()
        bkhat = mask(reverse=True)
    except TypeError:
        pass

    if synthesis :
        print('Synthesizing')
        bkhat = TF.TimeFrequencyDecomposition.iSTFT(bkhat, pX, 1025, hop, True)
        svhat = TF.TimeFrequencyDecomposition.iSTFT(svhat, pX, 1025, hop, True)

    return svhat, bkhat

def estimate_multichannel_sources(mX, pX, hop):
    """ Estimate singing voice and background music using a trained
    supervised method based on GRU-denoising auto-encoding.
    Args:
        mX          : 	(3D ndarray) Mixture magnitude spectrum
        pX          :   (3D ndarray) Mixture phase spectrum
        hop         :   (int)        Hop size for STFT analysis & synthesis
                        Expected shape : (Channels x Frequency-samples x Time-frames)
    Returns:
        svhat       :   (2D ndarray) Time-domain waveform of the multichannel estimated singing voice
        bkhat       :   (2D ndarray) Time-domain waveform of the multichannel background/accompaniment

    """
    # Monaural estimation for each channel
    svL, bkL = estimate_sources(mX[0, :, :].T, pX[0, :, :].T, hop, maskingMode = 2, synthesis = False)
    svR, bkR = estimate_sources(mX[1, :, :].T, pX[1, :, :].T, hop, maskingMode = 2, synthesis = False)

    # Reshaping
    sv = np.zeros((2, svL.shape[1], svL.shape[0]), dtype = np.float32)
    bk = np.zeros((2, bkR.shape[1], bkR.shape[0]), dtype = np.float32)
    sv[0, :, :] = svL.T
    sv[1, :, :] = svR.T
    bk[0, :, :] = bkL.T
    bk[1, :, :] = bkR.T
    mX, pX = reshape_data(mX, pX)

    # Multi-channel Wiener filtering
    mask = fm(mX, np.sum(sv, axis = 0, keepdims = True), np.sum(bk, axis = 0, keepdims = True), [], [], alpha = 1.35, method = 'MWF')
    svhat = mask()

    # Synthesis
    svhat = TF.TimeFrequencyDecomposition.MCiSTFT(svhat, pX, 1025, hop, True)
    bkhat = TF.TimeFrequencyDecomposition.MCiSTFT(mX, pX, 1025, hop, True)
    bkhat -= svhat

    return svhat, bkhat

if __name__ == '__main__':
    # Parameters
    # A dictionary for calling various models and setting results paths.
    # Please select one of the following numbers :
    processesDict = {
        0 : ['/home/GRU_skip_res/IBM/', 0],   # GRU-DBM
        1 : ['/home/GRU_skip_res/aATF/', 1],  # GRU-DADM
        2 : ['/home/GRU_skip_res/aCTF/', 2],  # GRU-D
        3 : ['/home/GRU_skip_res/WTF/', 3],   # GRU-DWF
        4 : ['/home/GRU_skip_res/sACTF/', 4], # GRU-S
        5 : ['/home/GRU_skip_res/raw/', 5]    # Raw outputs
                    }
    savepath = processesDict[np.int(sys.argv[1])][0]
    mmode = processesDict[np.int(sys.argv[1])][1]

    print(savepath, mmode)

    hop = 256
    dim = 1025
    seqlen = 18
    overlap = 6
    w = sig.hamming(1025, True)
    multichannel = False

    # For demo purposes
    x, fs = IO.AudioIO.wavRead('testFiles/test_file.wav', mono = True)
    # Single-channel case
    mX, pX = TF.TimeFrequencyDecomposition.STFT(x, w, 2048, hop)
    # Multi-channel case
    #mX, pX = TF.TimeFrequencyDecomposition.MCSTFT(x, w, 2048, hop)
    # Estimating Sources
    svhat, bkhat = estimate_sources(mX, pX, hop, maskingMode = mmode)
    #svhat, bkhat = estimate_multichannel_sources(mX, pX, hop)
    print('Done!')

    # Evaluation
    if 'x' not in locals():
        SDR = []
        ISR = []
        SIR = []
        SAR = []
        for fileIndx in xrange(51, 101):            # Iterate over test sub-set
            # Check if multichannel case is necessary
            if multichannel:
                x, xsv, xbk, fs = crawlDSD(fileIndx, mono = False)
                print('Multichannel Analysis')
                mX, pX = TF.TimeFrequencyDecomposition.MCSTFT(x, w, 2048, hop)
                # Estimating Sources
                svhat, bkhat = estimate_multichannel_sources(mX, pX, hop)

                print('BSS Evaluation')
                # Preparing Data for Evaluation
                vX, vpX = TF.TimeFrequencyDecomposition.MCSTFT(xsv, w, 2048, hop)
                bX, bpX = TF.TimeFrequencyDecomposition.MCSTFT(xbk, w, 2048, hop)
                vX, vpX = reshape_data(vX, vpX)
                bX, bpX = reshape_data(bX, bpX)
                xsv = TF.TimeFrequencyDecomposition.MCiSTFT(vX, vpX, 1025, hop, True)
                xbk = TF.TimeFrequencyDecomposition.MCiSTFT(bX, bpX, 1025, hop, True)

                # Last sanity check
                if len(svhat) > len(xsv):
                    svhat = svhat[:len(xsv), :]
                    bkhat = bkhat[:len(xsv), :]
                else :
                    xsv = xsv[:len(svhat), :]
                    xbk = xbk[:len(svhat), :]

            else :
                x, xsv, xbk, fs = crawlDSD(fileIndx, mono = True)
                print('Analysis')
                mX, pX = TF.TimeFrequencyDecomposition.STFT(x, w, 2048, hop)
                # Estimating Sources
                svhat, bkhat = estimate_sources(mX, pX, hop, maskingMode = mmode)

                print('BSS Evaluation')
                # Preparing Data for Evaluation
                vX, vpX = TF.TimeFrequencyDecomposition.STFT(xsv, w, 2048, hop)
                bX, bpX = TF.TimeFrequencyDecomposition.STFT(xbk, w, 2048, hop)
                if gdae.trimframe == 0:
                    vX = vX[overlap/2:, :]
                    vpX = vpX[overlap/2:, :]
                    bX = bX[overlap/2:, :]
                    bpX = bpX[overlap/2:, :]
                else :
                    vX = vX[gdae.trimframe + overlap/2: - gdae.trimframe, :]
                    vpX = vpX[gdae.trimframe + overlap/2: - gdae.trimframe, :]
                    bX = bX[gdae.trimframe + overlap/2: - gdae.trimframe, :]
                    bpX = bpX[gdae.trimframe + overlap/2: - gdae.trimframe, :]

                xsv = TF.TimeFrequencyDecomposition.iSTFT(vX, vpX, 1025, hop, True)
                xbk = TF.TimeFrequencyDecomposition.iSTFT(bX, bpX, 1025, hop, True)

                if len(svhat) > len(xsv):
                    svhat = svhat[:len(xsv)]
                    bkhat = bkhat[:len(xsv)]
                else :
                    xsv = xsv[:len(svhat)]
                    xbk = xbk[:len(svhat)]

            print('Writing to disk')
            IO.AudioIO.audioWrite(svhat, fs, 16, os.path.join(savepath, 'svhat_'+str(fileIndx)+'.m4a'), 'm4a')  # Use wavWrite and '.wav' for Matlab-based evaluation
            IO.AudioIO.audioWrite(bkhat, fs, 16, os.path.join(savepath, 'bkhat_'+str(fileIndx)+'.m4a'), 'm4a')  # Use wavWrite and '.wav' for Matlab-based evaluation
            IO.AudioIO.audioWrite(xsv, fs, 16, os.path.join(savepath, 'svtrue_'+str(fileIndx)+'.m4a'), 'm4a')   # Use wavWrite and '.wav' for Matlab-based evaluation
            IO.AudioIO.audioWrite(xbk, fs, 16, os.path.join(savepath, 'bktrue_'+str(fileIndx)+'.m4a'), 'm4a')   # Use wavWrite and '.wav' for Matlab-based evaluation

            # In case that evaluation takes place in python (Matlab BSSEval-images was used for the paper)
            print('Evaluating')
            cSDR, cISR, cSIR, cSAR, _ = bssEval.bss_eval_images_framewise([xsv, xbk], [svhat, bkhat])
            SDR.append(cSDR)
            ISR.append(cISR)
            SIR.append(cSIR)
            SAR.append(cSAR)

            # Saving Results
            pickle.dump(SDR, open(os.path.join(savepath, 'SDR.p'), 'wb'))
            pickle.dump(ISR, open(os.path.join(savepath, 'ISR.p'), 'wb'))
            pickle.dump(SIR, open(os.path.join(savepath, 'SIR.p'), 'wb'))
            pickle.dump(SAR, open(os.path.join(savepath, 'SAR.p'), 'wb'))