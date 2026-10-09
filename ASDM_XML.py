# Module to read XML file in prefix
#
import os
import xml.etree.ElementTree as ET
import numpy as np
from interferometry import indexList
def CheckCorr( prefix ):
    if not os.path.isdir(prefix): return 'ALMA_BASELINE'  # Default BLCORR
    Corr_XML = prefix + '/CorrelatorMode.xml'
    tree = ET.parse(Corr_XML)
    root = tree.getroot()
    for row in root.findall('row'):
        #---- Check by Correlat name
        for CorrID in row.findall('correlatorName'): CorrName = CorrID.text
    #
    return CorrName
#
def BandList( prefix ):
    KEY_ALMA ='ALMA_RB'
    RXList = []
    #-------- Check Receiver bands
    RB_XML = prefix + '/' + 'Receiver.xml'
    tree = ET.parse(RB_XML)
    root = tree.getroot()
    for row in root.findall('row'):
        #-------- Find Receiver band name
        for RX in row.findall('frequencyBand'): RXname = RX.text
        if KEY_ALMA in RXname:  RXList = RXList + [RXname.replace(KEY_ALMA, 'RB')]
    #
    return list(set(RXList))
#
def spwMS(msfile):
    from casatools import msmetadata as msmdtool
    msmd = msmdtool()
    msmd.open(msfile)
    spwList = list(set(msmd.tdmspws()) | set(msmd.fdmspws()))
    spwList = [int(spw) for spw in spwList if len(msmd.scansforspw(spw)) > 0]
    spwList.sort()
    spwDic  = dict(zip(spwList, [[]]*len(spwList))) # Dictionary for spw info
    for spw in spwDic.keys():
        chFreq = msmd.chanfreqs(spw)
        spwDic[spw] = {
            'Band'   : [spwName.split('ALMA_')[1] for spwName in msmd.namesforspws(spw)[0].split('#') if 'ALMA_' in spwName][0],
            'BB'     : msmd.baseband(spw),
            'chNum'  : len(chFreq),
            'refFreq': msmd.reffreq(spw)['m0']['value'],
            'BW'     : msmd.bandwidths(spw),
            'ch0'    : chFreq[0],
            'chStep' : np.median(msmd.chanwidths(spw)),
            'refChan': -0.5,
            'scanList': msmd.scansforspw(spw).tolist()
        }
    msmd.close()
    return spwDic
#
