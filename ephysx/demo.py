#!/usr/bin/python3

import daw.ephysx.pyphy as pyphy

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

