# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:05:36 2015

@author: Pol Delgado Martin
"""
import os
import re
from numpy import *
from matplotlib.pylab import *
from scipy.signal import *
import scipy as sp
from sklearn import mixture
from scipy.io import wavfile

songs=filter(lambda x: x if re.search('wav',x) else None, os.listdir('audio'))

for song in songs:
    rate,data=wavfile.read('audio/'+song)
    
    F=fft(data)
    F=F[range(len(F)/2)]
    plt.plot(decimate(F[0:len(F)/256],2))
    g=mixture.GMM(n_components=9)
    g.fit(F)
    
    aux=linspace(F.min(),F.max(),1000)
    
        
    plt.plot(g.score_samples(aux)[0])
    

        
    

    data=random(500)*200 + 700
    
    g=mixture.GMM(1)
    g.fit(data)
    
    logprob, responsibilities = g.score_samples(x)
    pdf = np.exp(logprob)
    plot(pdf)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    wavfile.write('out.wav',rate,data)
    
    


