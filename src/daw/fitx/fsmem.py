#!/usr/bin/python3

import numpy as np

# See matlab/octave version at ~/octave/dw/apps/fsmem/fsmem1d for details

class MOGFSMEM1D:
    def __init__(self, K, epsi=.001, lmbd=.01):
        '''K: nr of clusters
        epsi: relative change of log likelihood used for termination
        lambda: fudge parameter to prevent zero-variance attractor'''
        self.K = K
        self.epsi = epsi
        self.lmbd = lmbd

        self.max_iter = 100
        self.fudge = 1e-6
        self.max_cands = 5
        self.split_init_epsi = 1
        self.p = None
        self.mu = None
        self.sig = None
        self.R = None

    def __repr__(self):
        return f"p={self.p}\nmu={self.mu}\nsig={self.sig}"

    
    def init(self, x):
        self.x = x
        self.N = len(x)
        p = np.ones(self.K) / self.K
        idx = (np.random.random(self.K)*self.N).astype(int)
        mu = x[idx]
        datvar = np.var(x)
        sig = np.repeat(datvar/40, self.K)
        self.p = p
        self.mu = mu
        self.sig = sig


    def responsibility(self, p, mu, sig):
        K = len(p)
        norma = np.zeros(self.N)
        R = np.zeros((self.N, K))
        px = np.zeros(self.N)
        nG = (2*np.pi)**(-1/2)

        for k in range(K):
            s = sig[k]
            siginv = 1/s
            detsig = s
            dx = self.x - mu[k]
            sdx = siginv * dx
            expo = -.5 * dx*sdx
            G = nG / np.sqrt(detsig) * np.exp(expo)
            pG = p[k] * G
            px += pG
            norma += pG
            R[:,k] = pG

        for k in range(K):
            R[:,k] /= (norma + 1e-300)

        likely = np.sum(np.log(px + 1e-300))
            
        return R, likely


    def mergemerit(self, R):
        return np.matmul(R.T, R)


    def splitmerit(self, R):
        K = len(self.p)
        J = np.zeros(K)
        nG = (2*np.pi)**(-1/2)

        for k in range(K):
            s = self.sig[k]
            siginv = 1/s
            detsig = s
            dx = self.x - self.mu[k]
            sdx = siginv * dx
            expo = -.5 * dx*sdx
            G = nG / np.sqrt(detsig) * np.exp(expo)

            f = R[:,k] / (np.sum(R[:,k]) + self.fudge)
            G += .00001
            idx = f > .00001
            J[k] = np.sum(f[idx] * np.log(f[idx] / G[idx]))

        return J


    def merits(self):
        K = len(self.p)
        R, _ = self.responsibility(self.p, self.mu, self.sig)
        Jm = self.mergemerit(R)
        Jsplit = self.splitmerit(R)
        Jmerge = np.zeros((K*(K-1)//2, 3))
        idx = 0
        for k in range(K):
            for l in range(k+1,K):
                Jmerge[idx,:] = [ Jm[k,l], k, l ]
                idx += 1

        Jsplit = np.stack((Jsplit, np.arange(K)), 1)

        idx = np.argsort(-Jmerge[:,0])
        Jmerge = Jmerge[idx,:]
        idx = np.argsort(-Jsplit[:,0])
        Jsplit = Jsplit[idx,:]
        return Jmerge, Jsplit


    def partialem(self, p, mu, sig, idx):
        K = len(p)
        R,_ = self.responsibility(p, mu, sig)
        primalR = np.sum(R[:,idx], 1)
        old_likely = -1e9
        nG = (2*np.pi)**(-1/2)
        for it in range(self.max_iter):
            lastit = it

            norma = np.zeros(self.N)
            R = np.zeros((self.N, K))
            px = np.zeros(self.N)
            nG = (2*np.pi)**(-1/2)

            for k in idx:
                s = sig[k]
                siginv = 1/s
                detsig = s
                dx = self.x - mu[k]
                sdx = siginv * dx
                expo = -.5 * dx*sdx
                G = nG / np.sqrt(detsig) * np.exp(expo)
                pG = p[k] * G
                px += pG
                norma += pG
                R[:,k] = pG
            likely = np.sum(np.log(px + 1e-300))
            if np.abs((likely - old_likely) / likely) < self.epsi:
                break
            old_likely = likely

            norma = primalR / (norma + 1e-300)
            for k in idx:
                R[:,k] = R[:,k] * norma

            for k in idx:
                sumR = np.sum(R[:,k])
                mu[k] = np.sum(self.x*R[:,k]) / (sumR + self.fudge)
                dx = self.x - mu[k]
                dxdx = dx*dx
                Rsdxdx = np.sum(dxdx * R[:,k])
                sig[k] = (Rsdxdx + self.lmbd)/(sumR+self.lmbd)
                p[k] = np.mean(R[:,k])
            return p, mu, sig, lastit
        
    
    def fullem(self, p, mu, sig):
        '''MOSG_FULLEM implements the EM algorithm for mixture of Gaussians.
        Output: likely: log likelihood at end of run
        Algorithm: Max Welling, in class notes for CS156b
        Matlab Coding:    Daniel Wagenaar, April-May 2000.
        Python translation: DAW, September 2021'''
        K = len(p)

        old_likely = -1e9
        nG = (2*np.pi)**(-1/2)

        for it in range(self.max_iter):
            lastit = it

            #  E step: compute responsibilities
            R, likely = self.responsibility(p, mu, sig)

            if np.abs((likely - old_likely) / likely) < self.epsi:
                break
            old_likely = likely

            #  M step: recompute mu, sig, p
            for k in range(K):
                sumR = np.sum(R[:,k])
                mu[k] = np.sum(self.x*R[:,k]) / (sumR+self.fudge)
                dx = self.x - mu[k]
                dxdx = dx * dx
                Rsdxdx = np.sum(dxdx*R[:,k])
                sig[k] = (Rsdxdx + self.lmbd)/(sumR+self.lmbd);
            p = np.mean(R, 0)

        return p, mu, sig, likely - .5*np.log(self.N)*K*(1+1+1), lastit


    def mergeinit(self, m1, m2):
        p = self.p.copy()
        mu = self.mu.copy()
        sig = self.sig.copy()
        p[m1] += p[m2]
        mu[m1] = (mu[m1] + mu[m2]) / 2
        sig[m1] = (sig[m1] + sig[m2]) / 2
        p[m2] = 0
        return self.truncate(p, mu, sig, m2)


    def splitinit(self, p, mu, sig, k, l):
        p = p.copy()
        mu = mu.copy()
        sig = sig.copy()
        p[k] /= 2
        p[l] = p[k]
        sd = np.sqrt(sig[k])
        mu[l] = mu[k] + self.split_init_epsi * sd * np.random.randn(1)
        mu[k] += self.split_init_epsi * sd * np.random.randn(1)
        sig[l] = sig[k]
        return p, mu, sig

    
    def truncate(self, p, mu, sig, k0):
        K = len(p)
        p = np.delete(p, k0)
        mu = np.delete(mu, k0)
        sig = np.delete(sig, k0)
        return p, mu, sig


    def extend(self, p, mu, sig):
        p = np.append(p, 0)
        mu = np.append(mu, 0)
        sig = np.append(sig, 1)
        return p, mu, sig


    def trytomerge(self, Jmerge, fail):
        merge_cands = Jmerge.shape[0]
        if merge_cands > self.max_cands:
            merge_cands = self.max_cands
        likely = -1e9
        for c in range(merge_cands):
            m1 = int(Jmerge[c, 1])
            m2 = int(Jmerge[c, 2])
            p, mu, sig = self.mergeinit(m1, m2)
            p, mu, sig, it = self.partialem(p, mu, sig, [m1])
            self.iters.append(it)
            p, mu, sig, likely, it, = self.fullem(p, mu, sig)
            self.iters.append(it)

            if likely > self.likelies[-1]:
                break
        if likely > self.likelies[-1]:
            self.p, self.mu, self.sig = p, mu, sig
            self.likelies.append(likely)
            nxt = 1
            fail = .5
        else:
            fail += 1
            nxt = -1
        return nxt, fail


    def trytosplit(self, Jsplit, fail):
        likely = -1e99
        split_cands = Jsplit.shape[0]
        if split_cands > self.max_cands:
            split_cands = self.max_cands

        for c in range(split_cands):
            s = int(Jsplit[c,1])
            p, mu, sig = self.extend(self.p, self.mu, self.sig)
            s2 = len(p)-1
            p, mu, sig = self.splitinit(p, mu, sig, s, s2)
            p, mu, sig, it = self.partialem(p, mu, sig, [s, s2])
            self.iters.append(it)
            p, mu, sig, likely, it = self.fullem(p, mu, sig)
            self.iters.append(it)
            if likely > self.likelies[-1]:
                break

        if likely > self.likelies[-1]:
            self.p, self.mu, self.sig = p, mu, sig
            self.likelies.append(likely)
            fail = .5
            nxt = -1
        else:
            fail += 1
            nxt = 1
        return fail, nxt

    
    def fit(self, x):
        self.init(x)
        self.likelies = []
        self.iters = []

        self.p, self.mu, self.sig, likely, it \
            = self.fullem(self.p, self.mu, self.sig)
        self.likelies.append(likely)
        self.iters.append(it)

        nxt = 1
        fail = 0.5

        for it in range(self.max_iter):
            Jmerge, Jsplit = self.merits()

            if nxt>0:
                nxt, fail = self.trytomerge(Jmerge, fail)
            else:
                nxt, fail = self.trytosplit(Jsplit, fail)

            if fail>2:
                break

    def plot(self, xxx, scl=1):
        K = len(self.p)
        for k in range(K):
            yyy = np.exp(-.5*(xxx-self.mu[k])**2/self.sig[k])
            yyy *= scl * self.p[k] / np.sqrt(2*np.pi*self.sig[k]) * len(self.x)
            plt.plot(xxx, yyy)
        
if __name__== "__main__":                
    ######################################################################
    xx = np.array([ 364., 2117.,  338.,  872., 1489., 1952., 2756.,  368., 2228., 
           2992.,  257.,  541., 2116.,  314.,  382.,  357., 2055.,  366., 
            384.,  325.,  373.,  243.,  494.,  360.,   69.,  351., 1365., 
           1910., 2587.,  368.,  471.,  353., 2084.,   78.,  225.,   13., 
            396., 1344.,  491.,  895., 1116., 1704., 2348.,   62.,  526., 
           1161., 1636., 2181., 2706., 2038., 2660., 2972.,  318., 1168., 
           2204.,  354., 2609.,  330.,  339.,  272.,  359.,  303.,  270., 
           2419.,  351., 1968., 2461.,  500., 1054., 1653., 2444.,  322., 
           1092.,  498., 1412., 2772.,  229., 1276., 2011., 2366., 2632., 
            328.,  604., 2270., 2539., 2863.,  249., 2446., 2824.,  352., 
           2037.,  326., 2611.,  290.,  525., 1396., 1742., 2198., 2538., 
            300.,  677.,   23.,  370.,  360.,  366.,  416.,    6., 1041., 
            367.,  376., 2675.,  369.,  363.,  392.,  360.,  512.,  364., 
            352.,  359.,  344.,  449.,  498.,  360.,  362.,  367.,  337., 
            514.,  339.,  601.,  398.,  404.,  335.,  375.,  352.,  366., 
            419.,  487.,  369.,  447.,  329.,  345.,  375.,  421.,  352., 
            384., 2140.,  351.,  359.,  339.,  274.,  375.,  519.,  307., 
            340.,  281., 1009., 1979.,  512.,  362.,  330.,  423.,  369., 
            271.,  338., 1420., 2251., 2736.,  476.,  361.,  356.,  281., 
           1527.,  454.,  471.,  278.,  265., 1281., 1927., 2541.,  245., 
           2578.,  332.,  346.,  367.,  362.,  421.,  370.,  353.,  347., 
            499., 1820., 2896.,  252.,  301.,  574.,  447.,  338.,  357., 
            264., 2567.,  343.,  282.,  410.,  441.,  354.])/30 - 5;

    mog = MOGFSMEM(5)
    mog.fit(xx)
    print(mog)

    import matplotlib.pyplot as plt
    plt.interactive(True)
    plt.close('all')
    dt=.5
    plt.hist(xx, np.arange(-20,100,dt))
    mog.plot(np.arange(-20,100,.01),scl=dt)


