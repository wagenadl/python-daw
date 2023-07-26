#!/usr/bin/python3

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import time

class SelectSpikes:
    '''SELECTSPIKES - Manually select spikes based on graphical threshold
    ss = SELECTSPIKES(tms, hei) creates a GUI to manually select spikes
    using a draggable threshold. 
    idx = ss.run() runs the GUI and reports the result
    This is a re-implementation of the Matlab/Octave version I wrote a 
    decade ago.'''
    def __init__(self, tms, hei):
        self.t = tms
        self.y = hei
        self.t0 = tms[0]
        self.t1 = tms[-1]
        tblue = np.array([np.mean(tms)])
        tgreen = np.array([np.mean(tms)])
        self.tdots = [tblue, tgreen]
        yblue = np.array([np.mean(np.abs(hei))])
        ygreen = np.array([2*np.mean(np.abs(hei))])
        self.ydots = [yblue, ygreen]
        self.colors = ['g', 'b']
        self.clickthr = 10

        self.fig, self.ax = plt.subplots(figsize=[15, 5])
        self.hdata = self.ax.plot(self.t, self.y, 'r.')
        self.hlines = []
        self.hdots = []
        for k in range(2):
            self.hlines.append(self.ax.plot(np.hstack((self.t0,
                                                       self.tdots[k],
                                                       self.t1)),
                                            np.hstack((self.ydots[k][0],
                                                       self.ydots[k],
                                                       self.ydots[k][-1])),
                                            self.colors[k]))
            self.hdots.append(self.ax.plot(self.tdots[k], self.ydots[k],
                                           self.colors[k] + '.',
                                           markersize=20))

        self.draggingdot = None # Or tuple (k, idx)
        self.cidp = self.fig.canvas.mpl_connect('button_press_event',
                                               self.onpress)
        self.cidr = self.fig.canvas.mpl_connect('button_release_event',
                                               self.onrelease)
        self.cidm = None

    def selection(self):
        '''SELECTION - Indices of included spikes
        idx = SELECTION(ss) returns the indices of the spikes between the 
        lines.'''
        if np.mean(self.ydots[1]) < np.mean(self.ydots[0]):
            kk = [1, 0]
        else:
            kk = [0, 1]
        sgms_low = self.segments(kk[0])
        sgms_high = self.segments(kk[1])
        use = np.zeros(self.t.shape, dtype=bool)
        for n in range(len(self.t)):
            t = self.t[n]
            y = self.y[n]
            q, ylow = self.findsegment(t, sgms_low)
            q, yhigh = self.findsegment(t, sgms_high)
            if ylow is not None and yhigh is not None:
                use[n] = y>=ylow and y<=yhigh
        return np.nonzero(use)[0]
        
    def run(self, leaveopen=False):
        '''RUN - Show the figure and wait until clicked outside
        sel = RUN(ss) makes sure the figure is visible, lets the user interact
        with it until s/he clicks outside the axes area, then returns the
        selection. See also SELECTION.'''
        b = plt.isinteractive()
        plt.ion()
        plt.show()
        self.quit = False
        print('''Drag lines to enclose desired spikes:
Click on a line segments to add a new handle.
Drag a handle past another handle or edge of graph to make it go away.
Click in margin of figure to end interaction.
The lines are equivalent: it's OK for green to be above blue. 
However, the lines should not cross.''')

        while not self.quit:
            plt.pause(1)
        if not leaveopen:
            plt.close(ss.fig)
        if not b:
            plt.ioff()
        return self.selection()

    def pixdist(self, dt, dy):
        '''PIXDIST - Pixel distance given data distance
        r = PIXDIST(ss, dt, dy) returns the Euclidean pixel distance
        between two graphical element given data spacing of (dt, dy) 
        between them. DT and DY may be vectors, in which case R will
        be a vector as well.'''
        pixbox = self.ax.get_window_extent().bounds # l,t,w,h
        databox = self.ax.dataLim.bounds # l,t,w,h
        dt = pixbox[2]*dt/databox[2]
        dy = pixbox[3]*dy/databox[3]
        return np.sqrt(dt*dt + dy*dy)

    def nearestdot(self, t, y):
        '''NEARESTDOT - Identity of and distance to nearest dot
        k, idx, r = NEARESTBLUEDOT(ss, x, y) returns the indentity of 
        and the distance to the dot nearest to (t, y).'''
        idx = []
        dd = []
        for k in range(2):
            dd1 = self.pixdist(t - self.tdots[k], y - self.ydots[k])
            idx1 = np.argmin(dd1)
            idx.append(idx1)
            dd.append(dd1[idx1])
        k = np.argmin(dd)
        return (k, idx[k], dd[k])

    def segments(self, k):
        '''SEGMENTS - List of segments in the k-th line
        sgm = SEGMENTS(ss, k) returns a list of segments in the k-th line.
        SGM is a list of tuples (t0, t1, y0, dy); the segment is defined
        on the interval [t0, t1] by y(t) = y0 + dy * (t-t0).'''
        sgm = []
        sgm.append((self.t0, self.tdots[k][0], self.ydots[k][0], 0))
        for i in range(len(self.tdots[k]) - 1):
            t0 = self.tdots[k][i]
            t1 = self.tdots[k][i+1]
            y0 = self.ydots[k][i]
            y1 = self.ydots[k][i+1]
            sgm.append((t0, t1, y0, (y1-y0)/(t1-t0)))
        sgm.append((self.tdots[k][-1], self.t1, self.ydots[k][-1], 0))
        return sgm

    def findsegment(self, t, sgms):
        for q in range(len(sgms)):
            t0, t1, y0, dy = sgms[q]
            if t>=t0 and t<=t1:
                return q, y0 + dy*(t-t0)
        return None, None
    
    def nearestsegment(self, t, y):
        dd = []
        idx = []
        for k in range(2):
            sgms = self.segments(k)
            q, yat = self.findsegment(t, sgms)
            if q is None:
                idx1 = None
                dd1 = np.inf
            else:
                idx1 = q
                dd1 = self.pixdist(0, y-yat)
            dd.append(dd1)
            idx.append(idx1)
        k = np.argmin(dd)
        return (k, idx[k], dd[k])

    def updatedotsandlines(self, k):
        self.hdots[k][0].set_data(self.tdots[k], self.ydots[k])
        self.hlines[k][0].set_data(np.hstack((self.t0,
                                              self.tdots[k],
                                              self.t1)),
                                   np.hstack((self.ydots[k][0],
                                              self.ydots[k],
                                              self.ydots[k][-1])))
        self.fig.canvas.draw()
        
    def onpress(self, event):
        t = event.xdata
        y = event.ydata
        if t is None or y is None:
            # click outside
            self.quit = True
            return
        
        k, idx, dd = self.nearestdot(t, y)
        if dd < self.clickthr:
            self.draggingdot = [k, idx]
        else:
            k, idx, dd = self.nearestsegment(t, y)
            if dd < self.clickthr:
                # Add a new dot
                self.tdots[k] = np.insert(self.tdots[k], idx, t)
                self.ydots[k] = np.insert(self.ydots[k], idx, y)
                self.updatedotsandlines(k)
                self.draggingdot = [k, idx]
        if self.draggingdot is not None:
            self.cidm = self.fig.canvas.mpl_connect('motion_notify_event',
                                                    self.onmove)
            

    def onrelease(self, event):
        if self.draggingdot is None:
           return

        k, idx = self.draggingdot
        self.draggingdot = None
        self.fig.canvas.mpl_disconnect(self.cidm)
        self.cidm = None

        if len(self.tdots[k]) <= 1:
            return
        
        drop = False
        if self.tdots[k][idx] <= self.t0:
            drop = True
        elif self.tdots[k][idx] >= self.t1:
            drop = True
        elif idx>0 and self.tdots[k][idx] <= self.tdots[k][idx-1]:
            drop = True
        elif idx+1 < len(self.tdots[k]) \
             and self.tdots[k][idx] >= self.tdots[k][idx+1]:
            drop = True
        if drop:
            self.tdots[k] = np.delete(self.tdots[k], idx)
            self.ydots[k] = np.delete(self.ydots[k], idx)
            self.updatedotsandlines(k)

    def onmove(self, event):
       if self.draggingdot is not None:
            k, idx = self.draggingdot
            t = event.xdata
            y = event.ydata
            self.tdots[k][idx] = t
            self.ydots[k][idx] = y
            self.updatedotsandlines(k)

def selectspikes(tms, hei, return_idx=False):
    '''SELECTSPIKES - Run the SelectSpikes GUI
    tms, hei = SELECTSPIKES(tms, hei) runs the GUI and returns times
    and amplitudes of selected spikes.
    idx = SELECTSPIKES(tms, hei, True) runs the GUI and returns the indices
    of selected spikes instead.'''
    ss = SelectSpikes(tms, hei)
    idx = ss.run()
    if return_idx:
        return idx
    else:
        return tms, hei
            
if __name__ == '__main__':
    import scipy.signal
    from ephysx import spikex
    import pyqplot as qp

    b, a = scipy.signal.butter(3, .1)
    N = 1000
    tt = np.arange(N) / 1e3
    rnd = np.random.randn(N)
    flt = scipy.signal.filtfilt(b, a, rnd)
    qp.figure('/tmp/s1')
    qp.pen('b', 1)
    qp.plot(tt, flt)
    idx = spikex.detectspikes(flt, .1, tkill=0)
    qp.pen('r')
    qp.marker('o', 2)
    qp.mark(tt[idx], flt[idx])
    id2 = spikex.cleancontext(idx, flt)
    qp.marker('o', 4, fill='open')
    qp.pen('g', 1)
    qp.mark(tt[id2], flt[id2])
    qp.pen('k', .5)
    qp.xaxis()
    qp.yaxis()
    qp.shrink()

    ss = SelectSpikes(tt[idx], flt[idx])
    print(ss.run())
