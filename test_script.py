import numpy as np
import convert_catalog as cc
import matplotlib.pyplot as plt

scatters = [0.2] #only this one for now
scatter = scatters[0]
#Values for the RMR for fox
alpha   = 0.77
M_pivot = 10**12.63 #Msun/h
M_min   = 10**11.24
z = 1.

inpath  = "test_catalog.npy"

conv = cc.Converter(scatter, M_min=M_min, M_pivot=M_pivot, alpha=alpha)
data = np.load(inpath)
Ms = data[:,3] #Mass in Msun/h
Nh = len(data)
Nd = len(data[0]) #x, y, z, M, Np
out = np.zeros((Nh, Nd+3)) #x, y, z, M, Np, lambda_true, lambda_real, lambda_obs
out[:, :Nd]  = data #copy in x, y, z, M, Np                                
out[:, Nd]   = conv.lambda_true(Ms)
out[:, Nd+1] = conv.lambda_true_realization(Ms)
count = 0
for ltr in out[:, Nd+1]:
    try:
        out[:, Nd+2] = conv.Draw_from_CDF(1, ltr, z)
    except ValueError:
        x = conv.cdf4interp
        y = conv.l_out_grid
        dx = x[1:] - x[:-1]
        print x
        inds = dx <= 0
        ind = np.argmin(dx)
        print dx[inds]
        print dx[ind-2:ind+2]

        plt.plot(x[:-1]-x[1:])
        plt.show()
        plt.plot(x, y)
        plt.show()
        exit()
    count += 1
    if count%1 == 0:
        print "Finished %d / %d"%(count, len(Ms))
