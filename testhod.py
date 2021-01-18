#!/home/wtluo/anaconda/bin/python2.7

import numpy as np
import matplotlib.pyplot as plt
import hodstat

fmock = '../../../mock/mock_f0.10_s02.dat'
Mr,zl,zmax,Vmax,lgMh,R,icen = np.loadtxt(fmock,unpack=True,comments='#')
Mrlim = np.linspace(-19.0,-22.0,50)

fsatmk= np.zeros(49)
fsatsm= np.zeros(49)
for i in range(49):
  ixcen     = (Mr<=Mrlim[i]) & (Mr>=Mrlim[i+1]) & (lgMh>=12.16) & (lgMh<=14.1) & (icen==1)
  ixsat     = (Mr<=Mrlim[i]) & (Mr>=Mrlim[i+1]) & (lgMh>=12.16) & (lgMh<=14.1) & (icen==0)
  fsatmk[i] = float(len(Mr[ixsat]))/float(len(Mr[ixcen])+len(Mr[ixsat]))

  hods      = hodstat.HODSTAT(Mrlim[i],Mrlim[i+1]) 
  struc     = hods.occupation(np.mean(lgMh[ixcen]))
  cen       = struc['Central']
  sat       = struc['Satellite']
  fsatsm[i] = sat/(cen+sat)
  print Mrlim[i]-0.06,fsatsm[i],fsatmk[i],np.mean(lgMh[ixcen]),np.mean(lgMh[ixsat])
  #print len(Mr[ixsat]),len(Mr[ixcen])
plt.plot(Mrlim[0:49]-0.06,fsatsm,'k-',lw=3,label='HOD')
plt.plot(Mrlim[0:49]-0.06,fsatmk,'k--',lw=3,label='Mock')
plt.ylim(0.001,0.8)
#plt.yscale('log')
plt.xlim(-19.0,-22.0)
plt.show()



