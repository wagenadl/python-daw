#!/usr/bin/python3

from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QPolygon
from PyQt5.QtCore import Qt, QRect, QPoint
import numpy as np
from . import openEphysIO
import os
import ctypes as ct
import sip

if os.name=='posix':
    pfx = 'lib'
else:
    pfx =''
libname = pfx + 'pyphyc'

try:
    pyphyc = np.ctypeslib.load_library(libname, os.path.dirname(__file__))
except:
    pyphyc = np.ctypeslib.load_library(libname, '.')

pyphyc.quickdraw.argtypes = [ct.c_int64,
                             ct.c_int64,
                             ct.c_int,
                             ct.c_int,
                             ct.c_int,
                             ct.c_int,
                             ct.c_int,
                             ct.c_float]


def sensiblestep(mx):
    '''dx = SENSIBLESTEP(mx) returns a sensible step size not much smaller
    than MX:

      1<=MX<2  -> DX=1
      2<=MX<5  -> DX=2
      5<=MX<10 -> DX=5
    etc.'''
    
    lg=np.log10(mx)
    ord=np.floor(lg)
    sub = 10**(lg-ord)
    if sub>5:
        sub=5
    elif sub>2:
        sub=2
    else:
        sub=1
    return sub * 10**ord


class EPhysView(QWidget):
    def __init__(self, parent=None):
        self.mem = None
        self.fs_Hz = None
        self.chlist = None
        self.tscale_s = .1
        self.csep_digi = None
        self.cscale_chans = 0
        self.t0_s = 0
        self.c0 = 0
        self.margin_left = 60
        self.margin_top = 20
        self.margin_bottom = 20
        self.ticklen = 3
        self.tstim_s = []
        self.stimlabels = []
        super().__init__(parent)
        self.setWindowTitle('PyPhy')

    def setData(self, mem, fs_Hz, chlist=None):
        '''SETDATA - Specify data to display
        SETDATA(mem, fs_Hz) specifies the data (shaped TxC) to display and
        the sample rate for the data.
        Optional argument CHLIST must be a list or dict with channel numbers.
        Each value in CHLIST must be a dict with channel info as contained
        in OpenEphys's metadata. The only key required here is CHANNEL_NAME.
        '''
        self.mem = mem
        self.fs_Hz = fs_Hz
        self.chlist = chlist
        self.c0 = 0
        self.cscale_chans = min(mem.shape[1], 10)
        T = min(self.mem.shape[0], 65536)
        self.csep_digi = np.median(np.std(self.mem[:T,:], 0)) * 10
        self.update()

    def setStimuli(self, tt_s, labels=None):
        '''SETSTIMULI - Add stimulus markers to the display
        SETSTIMULI(tt) where TT is a vector of times in seconds, adds
        stimulus markers to the display.
        SETSTIMULI(tt, labels) where LABELS is a list of labels that
        has the same length as TT also provides labels for the stimuli.'''
        
        self.tstim_s = tt_s
        self.stimlabels = labels
        self.update()

    def wheelEvent(self, evt):
        if self.mem is None:
            return
        delta = evt.pixelDelta()
        dx = -delta.x()
        dy = -delta.y()
        C = self.mem.shape[1]
        T = self.mem.shape[0] / self.fs_Hz
        self.c0 = int(max(min(self.c0 + dy/2, C - self.cscale_chans), 0))
        self.t0_s = max(min(self.t0_s + dx/10 * self.tscale_s,
                            T - self.tscale_s), 0)
                            
        print(dx, dy)
        self.update()

    def keyPressEvent(self, evt):
        k = evt.key()
        m = evt.modifiers()
        if m==Qt.ShiftModifier:
            scl = 5
        elif m==Qt.ControlModifier:
            scl = 25
        elif m==Qt.AltModifier:
            scl = .2
        else:
            scl = 1
        if k==Qt.Key_Plus or k==Qt.Key_Equal:
            self.csep_digi /= np.sqrt(2)
        elif k==Qt.Key_Minus:
            self.csep_digi *= np.sqrt(2)
        elif k==Qt.Key_Period:
            self.t0_s = self.t0_s + self.tscale_s/2
            self.tscale_s /= 2
            self.t0_s = max(0, self.t0_s - self.tscale_s/2)
        elif k==Qt.Key_Comma:
            self.t0_s = self.t0_s + self.tscale_s/2
            self.tscale_s *= 2
            self.t0_s = max(0, self.t0_s - self.tscale_s/2)
        elif k==Qt.Key_BracketRight:
            self.c0 = self.c0 + self.cscale_chans//2
            self.cscale_chans = max(10, int(self.cscale_chans/2))
            self.c0 = max(0, self.c0 - self.cscale_chans//2)
        elif k==Qt.Key_BracketLeft:
            self.c0 = self.c0 + self.cscale_chans//2
            self.cscale_chans = min(self.mem.shape[1], 100,
                                    int(self.cscale_chans*2))
            self.c0 = max(0, self.c0 - self.cscale_chans//2)
        elif k==Qt.Key_PageUp or k==Qt.Key_Up:
            self.c0 = max(0, int(self.c0 - scl*self.cscale_chans))
        elif k==Qt.Key_PageDown or k==Qt.Key_Down:
            self.c0 = max(0, min(int(self.c0 + scl*self.cscale_chans),
                                 self.mem.shape[1]-self.cscale_chans))
        elif k==Qt.Key_Left:
            self.t0_s = max(0, self.t0_s - scl*self.tscale_s)
        elif k==Qt.Key_Right:
            self.t0_s = max(0, min(self.t0_s + scl*self.tscale_s,
                                   self.mem.shape[0]/self.fs_Hz
                                   - self.tscale_s))
        elif k==Qt.Key_T:
            t,ok = QInputDialog.getDouble(None, 'Go to time:',
                                       '(seconds)',
                                       self.t0_s + self.tscale_s/2,
                                       self.tscale_s/2,
                                       self.mem.shape[0]/self.fs_Hz
                                       - self.tscale_s/2,
                                       3)
            if ok:
                self.t0_s = max(0, min(t - self.tscale_s/2,
                                       self.mem.shape[0]/self.fs_Hz
                                       - self.tscale_s))
        elif k==Qt.Key_P:
            t0 = self.t0_s
            t1 = None
            i = 0
            for t in self.tstim_s:
                if t<t0:
                    t1 = t
                else:
                    break
                i += 1
            if t1 is not None:
                self.t0_s = max(0, t1 - self.tscale_s/4)
                lbl = f'Stimulus #{i} at {t1:.3f}'
                if self.stimlabels is not None:
                    lbl += f' “{self.stimlabels[i]}”'
                self.setWindowTitle(lbl)
        elif k==Qt.Key_N:
            t0 = self.t0_s + self.tscale_s/2
            t1 = None
            i = 0
            for t in self.tstim_s:
                if t>t0:
                    t1 = t
                    break
                i += 1
            if t1 is not None:
                self.t0_s = max(0, t1 - self.tscale_s/4)
                self.setWindowTitle(f'Stimulus #{i} at {t1:.3f}')
        elif k==Qt.Key_S:
            i,ok = QInputDialog.getInt(None, 'Go to stimulus:',
                                          '(#)',
                                          0,
                                          0,
                                          len(self.tstim_s))
            if ok:
                t1 = self.tstim_s[i]
                self.t0_s =  max(0, t1 - self.tscale_s/4)
                self.setWindowTitle(f'Stimulus #{i} at {t1:.3f}')
                
        else:
            return
        self.update()

    def paintEvent(self, evt):
        ptr = QPainter(self)
        try:
            ptr.fillRect(QRect(0,0,self.width(),self.height()),
                         QColor(0,0,0))
            w = self.width() - self.margin_left
            h = self.height() - self.margin_top - self.margin_bottom
            if w<5 or h<5:
                return
            ttick = sensiblestep(self.tscale_s/(1 + w/200))
    
            def t2x(t):
                return int(self.margin_left + (t-self.t0_s)*w/self.tscale_s + .5)
            def kk2x(kk):
                return self.margin_left + (kk/self.fs_Hz-self.t0_s)*w/self.tscale_s
    
            def cv2y(c, v=0):
                return int(self.margin_top + h*(c+.5-self.c0)/self.cscale_chans
                           + h*v/self.cscale_chans/self.csep_digi + .5)
            def cvv2y(c, v=0):
                return (self.margin_top + h*(c+.5-self.c0)/self.cscale_chans
                        + h*v/self.cscale_chans/self.csep_digi)
    
            t0 = np.ceil(self.t0_s/ttick) * ttick
            t1 = np.floor((self.t0_s+self.tscale_s)//ttick) * ttick
            ptr.setPen(QColor(255, 255, 255))
            for t in np.arange(t0, t1, ttick):
                x = t2x(t)
                ptr.drawText(QRect(x-100, 0,
                                   200, self.margin_top),
                             Qt.AlignCenter,
                             f'{t:.3f}')
                ptr.drawText(QRect(x-100, self.margin_top+h,
                                   200, self.margin_bottom),
                             Qt.AlignCenter,
                             f'{t:.3f}')
    
            ptr.setPen(QColor(64, 64, 64))
            for t in np.arange(t0, t1, ttick):
                x = t2x(t)
                ptr.drawLine(QPoint(x,self.margin_top),
                             QPoint(x,self.margin_top+h))
    
            ptr.setPen(QColor(255, 255, 255))
            for c in np.arange(self.c0, self.c0+self.cscale_chans, dtype=int):
                y = cv2y(c)
                if self.chlist is None:
                    cname = f'{c}'
                else:
                    cname = self.chlist[c]['channel_name']
                ptr.drawText(QRect(0, y-50,
                                   self.margin_left - 5, 100),
                             Qt.AlignRight + Qt.AlignVCenter,
                             cname)

            L = self.mem.shape[0]
            K = int(self.tscale_s * self.fs_Hz)
            k0 = min(int(self.t0_s * self.fs_Hz), L)
            k1 = min(k0 + K, L)
            #dat = np.zeros((2*K,), dtype=np.int32)
            #print(K, dat.shape)
            #dat[::2] = kk2x(np.arange(k0,k1)) + .5
            #for c in np.arange(self.c0, self.c0+self.cscale_chans, dtype=int):
            #    dat[1::2] = cvv2y(c, self.mem[k0:k1,c]) + .5
            #    ptr.drawPolyline(dat.data.tobytes(), K)
            C = self.mem.shape[1]
            c1 = min(self.c0 + self.cscale_chans, C)
            for c in np.arange(self.c0, c1, dtype=int):
                clr = QColor(int(127+127*np.cos(c*2.127+.1238)),
                             int(127+127*np.cos(c*4.5789+4.123809)),
                             int(127+127*np.cos(c*8.123+23.32412)))
                ptr.setBrush(clr)
                ptr.setPen(clr)
                dt1 = self.mem[k0:k1,c]
                if dt1.dtype!=np.int16:
                    print('convert')
                    dt1 = dt1.astype(np.int16)
                dataptr = dt1.__array_interface__['data'][0]
                datastride = (dt1[1:].__array_interface__['data'][0] - dataptr) // 2
                pyphyc.quickdraw(sip.unwrapinstance(ptr),
                                 dataptr,
                                 K,
                                 datastride,
                                 t2x(self.t0_s),
                                 w,
                                 cv2y(c),
                                 -h/self.cscale_chans/self.csep_digi) 
        finally:
            del ptr

            

if __name__=='__main__':
    root = '/media/wagenaar/datatransfer/2020-10-13_20-24-19'
    mem, s0, f_Hz, chlist = openEphysIO.loadcontinuous(root,
                                                       1,
                                                       1,
                                                       'Neuropix-PXI-101.0',
                                                       'salpa')
    pegs = np.loadtxt(f'{root}/experiment1/recording1/pegs.txt')
    app = QApplication([])
    wdg = EPhysView()
    wdg.setData(mem, f_Hz, chlist)
    wdg.setStimuli(pegs / f_Hz)
    wdg.show()
    app.exec()
    
