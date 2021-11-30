import daw.multitaper
%aimport daw.multitaper
import numpy as np
import matplotlib.pyplot as plt
from drepr import d
import scipy.signal
plt.interactive(True)

xx = np.random.randn(2000, 300)
f, pxx = daw.multitaper.pds(xx, nw=10, indiv=True)
T,R,K = pxx.shape
px = np.mean(pxx, -1)

plt.clf()
plt.plot(f, np.mean(px, -1))

fy, py = scipy.signal.welch(xx, axis=0)
plt.plot(fy, py.mean(-1))

#%%

pxx = pxx[1:-1,:,:]
plt.clf()
plt.plot(pxx.mean(-1).mean(-1))

#%%

px = pxx.mean(-1) # Average over tapers
px0 = px.mean(-1) # Average over experiments
px1 = px.mean(0) # Average over frequencies

plt.clf()
zz = np.arange(1.8, 2.2, .01)
yy0, _ = np.histogram(px0, zz)
yy1, _ = np.histogram(px1, zz)
plt.plot(yy0/yy0.sum())
plt.plot(yy1/yy1.sum())

#%%

s0 = np.mean(px.std(-1)) # Spread over experiments, avg over time
s1 = np.mean(pxx.std(-1)) # spread over tapers, avg over time and expts

print(s0, s1/np.sqrt(K))

#%%

ss0 = []
ss1 = []

for q in range(50):
    xx = np.random.randn(2000, 300)
    f, pxx = daw.multitaper.pds(xx, nw=10, indiv=True)
    pxx = pxx[1:-1,:,:]
    T,R,K = pxx.shape
    px = np.mean(pxx, -1)
    s0 = np.mean(px.var(-1)) # Spread over experiments, avg over time
    s1 = np.mean(pxx.var(-1)) # spread over tapers, avg over time and expts
    ss0.append(s0)
    ss1.append(s1)
    print(s0, s1/(K-1))

# Conclusion: The variance across tapers, divided by K-1, is the
# variance across experiments after averaging across tapers.

#%%

y,x = np.histogram(pxx.flatten(), np.arange(0,20,.01))
x = (x[:-1]+x[1:])/2

plt.clf()
plt.plot(x, y)

from daw.fitx import physfit

use = y>0
p = physfit('exp', x[use], y[use])
plt.plot(x, p[0].apply(x))

# Conclusion: The distribution of Pxx is e^-(Pxx/2).

xx = 4*np.random.randn(2000, 300)
f, pxx = daw.multitaper.pds(xx, nw=10, indiv=True)
pxx = pxx[1:-1,:,:]
    
y,x = np.histogram(pxx.flatten(), np.arange(0,20,.01))
x = (x[:-1]+x[1:])/2
p = physfit('exp', x[use], y[use])

# Corrected conclusion: P(Pxx) is e^-(Pxx/mean(Pxx)).

xx = np.random.randn(2000, 300)
f, pxx = daw.multitaper.pds(xx, nw=10, indiv=True)
pxx = pxx[1:-1,:,:]
y,x = np.histogram(np.log(pxx.flatten()), np.arange(0,20,.01))
x = (x[:-1]+x[1:])/2
plt.clf()
plt.plot(x, y)

from scipy.stats import expon

e = expon.fit(pxx.flatten())

#%%

tt = np.arange(1000).reshape(1000,1)/200
f = np.random.randn(1000).reshape(1,1000)*1 + 10
#f = 10
ph = np.random.randn(1000).reshape(1,1000)
xx = np.random.randn(1000, 1000) + 10*np.cos(2*np.pi*f*tt+2*np.pi*ph)
plt.clf()
plt.plot(tt, np.mean(xx,1))
f, pxx = daw.multitaper.pds(xx, f_s=200, f_res=2, indiv=True)
K = pxx.shape[-1]
pxx = pxx[1:-1,:,:]
f = f[1:-1]
plt.clf()
plt.plot(f, pxx.mean(-1).mean(-1))

px1 = pxx[np.argmin((f-10)**2),:,:]
px0 = pxx[np.argmin((f-75)**2),:,:]
X1 = np.median(px1)
X0 = np.median(px0)
y1,x1 = np.histogram(px1.flatten(), np.arange(0,5,.01)*X1)
y0,x0 = np.histogram(px0.flatten(), np.arange(0,5,.01)*X0)
x0 = (x0[:-1]+x0[1:])/2
x1 = (x1[:-1]+x1[1:])/2

plt.close('all')
_,ax=plt.subplots(2,1)
ax[0].plot(x1, y1)
ax[1].plot(x0, y0)

#%% Conclusion: Previous conclusion is junk. I cannot combine across frequency

_,ax=plt.subplots(2,1)
for k in range(K):
  px1 = pxx[np.argmin((f-10)**2),:,k]
  px0 = pxx[np.argmin((f-75)**2),:,k]
  X1 = np.median(px1)
  X0 = np.median(px0)
  y1,x1 = np.histogram(px1.flatten(), np.arange(0,50,.01))
  y0,x0 = np.histogram(px0.flatten(), np.arange(0,50,.01))
  x0 = (x0[:-1]+x0[1:])/2
  x1 = (x1[:-1]+x1[1:])/2
  
  ax[0].plot(x1, y1)
  ax[1].plot(x0, y0)
  
