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
