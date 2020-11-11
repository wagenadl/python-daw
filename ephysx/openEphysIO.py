#!/usr/bin/python3

import numpy as np
import ast



def loadoebin(exptroot, expt=1, rec=1):
    '''
    LOADOEBIN - Load the Open Ephys oebin file.  Oebin is a JSON file detailing
    channel information, channel metadata and event metadata descriptions.
    Contains a field for each of the recorded elements detailing their folder
    names, samplerate, channel count and other needed information.
    Parameters
    ----------
    exptroot : string
        Path to the folder of the general experiment.
    expt : integer, default is 1
        The subexperiment number.
    rec : integer, default is 1
        The recording number.
    Returns
    -------
    oebin : dictionary
        Organized information from oebin file into a dictionary for the selected
        subexperiment and recording.
    '''

    fldr = f'{exptroot}/experiment{expt}/recording{rec}'
    oebin = ast.literal_eval(open(f'{fldr}/structure.oebin').read())
    return oebin



def streaminfo(exptroot, expt=1, rec=1, section='continuous', stream=0):
    '''
    STREAMINFO - Return a dictionary containing the information session
    about the selected stream from the oebin file.
    Parameters
    ----------
    exptroot : string
        Path to the folder of the general experiment.
    expt : integer, default is 1
        The subexperiment number.
    rec : integer, default is 1
        The recording number.
    section : string, default is 'continuous'
        The name of the file
    stream : integer or string, default is 0
        The continuous data source we are getting the filename for.
    Returns
    -------
        Return a dictionary containing the information session about the
        indicated stream from the oebin file.
    Notes
    -----
    Optional argument STREAM specifies which continuous data to load. Default is
        index 0. Alternatively, STREAM may be a string, specifying the
        subdirectory, e.g., "Neuropix-PXI-126.1/". This method is preferred,
        because it is more robust. Who knows whether those stream numbers are
        going to be preserved from one day to the next.
        (The final slash is optional.)
    '''

    oebin = loadoebin(exptroot, expt, rec)
    if type(stream)!=str:
        return oebin[section][stream]
    if not stream.endswith('/'):
        stream += '/'
    for n in range(len(oebin[section])):
        if oebin[section][n]['folder_name'].lower() == stream.lower():
            return oebin[section][n]
    raise ValueError(f'Could not find {section} stream "{stream}"')



def contfilename(exptroot, expt=1, rec=1, stream=0, infix='continuous'):
    '''
    CONTFILENAME - Return the filename of the continuous data for a given recording.
    Parameters
    ----------
    exptroot : string
        Path to the folder of the general experiment.
    expt : integer, default is 1
        The subexperiment number.
    rec : integer, default is 1
        The recording number.
    stream : integer or string, default is 0
        The continuous data source we are getting the filename for.
    Returns
    -------
    ifn : string
        Full filename of the continuous.dat file.
    tsfn : numpy.ndarray
        Timestamps for the slected recording in the selected subexperiment.
    info : string
        The corresponding "continfo" section of the oebin file.
    Notes
    -----
    Optional argument STREAM specifies which continuous data to load. Default is
        index 0. Alternatively, STREAM may be a string, specifying the
        subdirectory, e.g., "Neuropix-PXI-126.1/". This method is preferred,
        because it is more robust. Who knows whether those stream numbers are
        going to be preserved from one day to the next.
        (The final slash is optional.)
    '''

    fldr = f'{exptroot}/experiment{expt}/recording{rec}'
    continfo = streaminfo(exptroot, expt, rec, 'continuous', stream)
    subfldr = continfo['folder_name']
    ifn = f'{fldr}/continuous/{subfldr}/{infix}.dat'
    tsfn = f'{fldr}/continuous/{subfldr}/timestamps.npy'
    return ifn, tsfn, continfo



def _doloadcontinuous(contfn, tsfn, continfo):
    '''
    _DOLOADCONTINUOUS - Load continuous data from a file.
    dat, s0, f_Hz, chinfo = DOLOADCONTINOUS(contfn, tsfn, continfo)
    performs the loading portion of the LOADCONTINUOUS function.
    '''

    mm = np.memmap(contfn, dtype=np.int16, mode='r') # in Windows os "mode=c" do not work 
    tms = np.load(tsfn, mmap_mode='r')
    C = continfo['num_channels']
    N = len(mm) // C
    chlist = continfo['channels']
    f_Hz = continfo['sample_rate']
    dat = np.reshape(mm, [N,C])
    s0 = tms[0]
    del mm
    return (dat, s0, f_Hz, chlist)



def loadcontinuous(exptroot, expt=1, rec=1, stream=0, infix='continuous', contfn=None):
    '''
    LOADCONTINUOUS - Load continuous data from selected data source in an
    Open Ephys experiment.
    Parameters
    ----------
    exptroot : string
        Path to the folder of the general experiment.
    expt : integer, default is 1
        The subexperiment number.
    rec : integer, default is 1
        The recording number.
    stream : integer or string, default is 0
        The continuous data source we are getting the filename for.
    contfn : null or string, default is None
    Returns
    -------
        Returns the outputs of _doloadcontinuous, i.e.
    dat : numpy.ndarray
        Data for the selected experiment and recording.
    s0 : numeric
        Sample number relative to the start of the experiment of the start of
        this recording.
    f_Hz : integer
        The sampling rate (in Hz) of the data set.
    chlist : list
        Channel information dicts, straight from the oebin file.
    Notes
    -----
    The returned value S0 is important because event timestamps are relative
        to the experiment rather than to a recording.
    Optional argument STREAM specifies which continuous data to load. Default is
        index 0. Alternatively, STREAM may be a string, specifying the
        subdirectory, e.g., "Neuropix-PXI-126.1/". This method is preferred,
        because it is more robust. Who knows whether those stream numbers are
        going to be preserved from one day to the next.
        (The final slash is optional.)
    Optional argument CONTFN overrides the name of the continuous.dat file,
        which is useful if you want to load the output of a preprocessed file
        (see, e.g., applyCAR).
    '''
    ourcontfn, tsfn, continfo = contfilename(exptroot, expt, rec, stream, infix)
    if contfn is not None:
        ourcontfn = contfn
    return _doloadcontinuous(ourcontfn, tsfn, continfo)



###################################################################################
###################################################################################
###################################################################################



def loadevents(exptroot, s0=0, expt=1, rec=1, stream=0, ttl='TTL_1'):
    '''
    LOADEVENTS - Load events associated with a continuous data stream.
    Parameters
    ----------
    exptroot : string
        Path to the folder of the general experiment.
    s0 :  integer, default is 0
        The first timestamp when hit play on Open Ephys gui. It  must
        be obtained from LOADCONTINUOUS.
    expt : integer, default is 1
        The subexperiment number.
    rec : integer, default is 1
        The recording number.
    stream : integer or string, default is 0
        The continuous data source we are getting the events for, either as an integer index or
        as a direct folder name.
    ttl : string, default is 'TTL_1'
        The TTL event stream that we are loading
    Returns
    -------
    ss_trl - s0 : numpy.ndarray
        The timestamps (samplestamps) of events (in samples) relative to the recording.
    bnc_cc : numpy.ndarray
        The event channel numbers associated with each event.
    bnc_states : numpy.ndarray
        Contains +/-1 indicating whether the channel went up or down.
    fw : numpy.ndarray
        The full_words 8-bit event states for the collection of events.
    Notes
    -----
    Optional argument STREAM specifies which continuous data to load. Default is
        index 0. Alternatively, STREAM may be a string, specifying the
        subdirectory, e.g., "Neuropix-PXI-126.1/". This method is preferred,
        because it is more robust. Who knows whether those stream numbers are
        going to be preserved from one day to the next.
        (The final slash is optional.)
    Note that SS_TRL can be used directly to index continuous data: Even though
        timestamps are stored on disk relative to start of experiment, this
        function subtracts the timestamp of the start of the recording to make life
        a little easier.
    '''

    fldr = f'{exptroot}/experiment{expt}/recording{rec}'
    evtinfo = streaminfo(exptroot, expt, rec, 'events', f'{stream}/{ttl}')
    subfldr = evtinfo['folder_name']
    ss_trl = np.load(f'{fldr}/events/{subfldr}/timestamps.npy')
    bnc_cc = np.load(f'{fldr}/events/{subfldr}/channels.npy')
    bnc_states = np.load(f'{fldr}/events/{subfldr}/channel_states.npy')
    fw = np.load(f'{fldr}/events/{subfldr}/full_words.npy')
    return (ss_trl - s0, bnc_cc, bnc_states, fw)



def filterevents(ss_trl, bnc_cc, bnc_states, channel=1, updown=1):
    '''
    FILTEREVENTS - Return only selected events from an event stream.
    Parameters
    ----------
    ss_trl : numpy.ndarray
        The samplestamps of events (in samples) relative to the recording.
    bnc_cc : numpy.ndarray
        The event channel numbers associated with each event.
    bnc_states : numpy.ndarray
        Contains +/-1 indicating whether the channel went up or down.
    channel : integer, default is 1
        The channel to use
    updown : integer, default is 1
        Set to -1 to extract the down events.
        Set to 0 to extract both the up and down events.
    Returns
    -------
    numpy.ndarray : The extracted timestamps for the up or down or both events for the
        selected channel.
        If updown is set to 0 also return a numpy.ndarray which is the extracted positive or negative answer.
    '''
    if updown==1:
        return ss_trl[np.logical_and(bnc_cc == channel, bnc_states > 0)]
    elif updown==-1:
        return ss_trl[np.logical_and(bnc_cc == channel, bnc_states < 0)]
    elif updown==0:
        return ss_trl[bnc_cc == channel], np.sign(bnc_states[bnc_cc==channel])
    else:
        raise ValueError('Bad value for updown')

def inferblocks(ss_trl, f_Hz, t_split_s=5.0):
    '''
    INFERBLOCKS - Split events into inferred stimulus blocks based on
    lengthy pauses.
    Parameters
    ----------
    ss_trl : numpy.ndarray
        The samplestamps of events (in samples) relative to the recording.
        (obtained from LOADEVENTS or FILTEREVENTS)
    f_Hz : integer
        Frequency (in Hz) of recording sampling rate.
    t_split_s : numeric, default is 5.0
    Returns
    -------
    ss_block : list
        List of numpy arrays samplestamps, one per block.
    Notes
    -----
    ss_block = INFERBLOCKS(ss_trl, f_Hz) splits the event time stamps SS_TRL (from LOADEVENTS
    or FILTEREVENTS) into blocks with cuts when adjacent events are more than 5 seconds
    apart. Optional argument T_SPLIT_S overrides that threshold.
    '''

    idx = np.nonzero(np.diff(ss_trl) >= t_split_s * f_Hz)[0] + 1
    N = len(ss_trl)
    idx = np.hstack((0, idx, N))
    ss_block = []
    for k in range(len(idx)-1):
        ss_block.append(ss_trl[idx[k]:idx[k+1]])
    return ss_block



def extractblock(dat, ss_trl, f_Hz, margin_s=10.0):
    '''
    EXTRACTBLOCK - Extract ephys data for a block of stimuli identified by SS_TRL
    which must be one of the items in the list returned by INFERBLOCKS.
    Parameters
    ----------
    dat : numpy.ndarray
        Ephys data from where we want to extract from.
    ss_trl : numpy.ndarray
        The samplestamps of event (in samples) relative to the recording which
        should be one of the items in the list returned by INFERBLOCKS.
    f_Hz : integer
        Frequency (in Hz) of recording sampling rate.
    margin_s : numeric, default is 10.0
        Length of the margin (in seconds) included at the beginning and end of
        the block (unless of course the block starts less than 10 s from the
        beginning of the file or analogously at the end).
    Returns
    -------
    dat[s0:s1,:] : numpy.ndarray
        Extracted portion of ephys data.
    ss_trl - s0 : numpy.ndarray
        Shifted timestamps of events (relative to the extracted portion of data).
    '''

    s0 = ss_trl[0] - int(margin_s * f_Hz)
    s1 = ss_trl[-1] + int(margin_s * f_Hz)
    S,C = dat.shape
    if s0 < 0:
        s0 = 0
    if s1 > S:
        s1 = S
    return dat[s0:s1,:], ss_trl - s0



def getbarcodes(ss_trl, bnc_cc, bnc_states, f_Hz, channel=1):
    '''
    GETBARCODES - Obtain barcodes from samplestamped rising and falling edges.
    Parameters
    ----------
    ss_trl : numpy.ndarray
        The samplestamps of an event (in samples) relative to the recording.
    bnc_cc : numpy.ndarray
        The event channel numbers associated with each event.
    bnc_states : numpy.ndarray
        Contains +/-(channel) indicating whether the channel went up or down.
    f_Hz : integer
        Frequency (in Hz) of recording sampling rate.
    channel : integer, default is 1
        The channel to use.
    Returns
    -------
    times :

    ss_barc :

    Notes
    -----

    '''
    ss_on = filterevents(ss_trl, bnc_cc, bnc_states)
    sss_on = inferblocks(ss_on, t_split_s=2, f_Hz=f_Hz)
    ss_off = filterevents(ss_trl, bnc_cc, bnc_states, updown=-1)
    sss_off = inferblocks(ss_off, t_split_s=2, f_Hz=f_Hz)
    N = len(sss_on)
    if len(sss_off) != N:
        raise Exception("Barcodes don't come in reasonable pairs")
    ss_barc = []
    times = []
    PERIOD = 120 * f_Hz / 30000
    for n in range(N):
        dton = sss_off[n] - sss_on[n]
        dtoff = sss_on[n][1:] - sss_off[n][:-1]
        dton = np.round(dton/PERIOD).astype(int)
        dtoff = np.round(dtoff/PERIOD).astype(int)
        value = 0
        K = len(dton)
        bit = 1
        for k in range(K):
            for q in range(dton[k]):
                value += bit
                bit *= 2
            if k<K-1:
                for q in range(dtoff[k]):
                    bit *= 2
        ss_barc.append(value)
        times.append(sss_on[n][0])
    return times, ss_barc



def matchbarcodes(ss1, bb1, ss2, bb2):
    '''
    MATCHBARCODES -
    Parameters
    ----------

    Returns
    -------

    Notes
    -----

    '''
    sss1 = []
    sss2 = []
    N1 = len(ss1)
    for n in range(N1):
        b = bb1[n]
        try:
            idx = bb2.index(b)
            sss1.append(ss1[n])
            sss2.append(ss2[idx])
        except:
            print(f'Caution: no match for barcode #{n}')
            pass # Barcode not matched
    return (sss1, sss2)


def aligntimestamps(ss_event_nidaq, ss_ni, ss_np):
    '''
    GETBARCODES -
    Parameters
    ----------

    Returns
    -------

    Notes
    -----

    '''

    ss_event_neuropix = np.interp(ss_event_nidaq, ss_ni, ss_np)
    # return ss_event_neuropix



def loadtranslatedevents(exptroot, expt=1, rec=1,
                         sourcestream='NI-DAQmx-142.0',
                         targetstream='Neuropix-PXI-126.0',
                         targetttl='TTL_1'):
    '''
    Return: as from loadevents, but ss_trl is translated to samples in the
    target stream.
    '''
    contfn, tsfn, info = contfilename(exptroot, expt, rec, stream=sourcestream)
    fs = info['sample_rate']
    (ss_trl, bnc_cc, bnc_states, fw) = loadevents(exptroot, s0=0, expt=expt, rec=rec, stream=sourcestream)
    t_ni, bc_ni = getbarcodes(ss_trl, bnc_cc, bnc_states, fs)
    (dat, s0, f_Hz, chlist) = loadcontinuous(exptroot, expt, rec, stream=targetstream)
    (ss1, cc1, vv1, fw1) = loadevents(exptroot, s0=s0, expt=expt, rec=rec, stream=targetstream, ttl=targetttl)
    t_np, bc_np = getbarcodes(ss1, cc1, vv1, f_Hz)
    ss_ni, ss_np = matchbarcodes(t_ni, bc_ni, t_np, bc_np)
    ss_trl = np.interp(ss_trl, ss_ni, ss_np)
    idx = np.nonzero(bnc_cc != 1)
    ss_trl = ss_trl[idx]
    bnc_cc = bnc_cc[idx]
    bnc_states = bnc_states[idx]
    fw = fw[idx]
    return ss_trl.astype(int), bnc_cc, bnc_states, fw
