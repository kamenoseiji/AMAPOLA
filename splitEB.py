import os
import numpy as np
from interferometry import GetPHSPWs, GetPHchavSPWs
from ASDM_XML import spwMS
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-u', dest='prefixList', metavar='prefixList',
    help='EB UID   e.g. uid___A002_X1399105_X40ca,uid___A002_X1399105_X414b,uid___A002_X1399105_X41c8', default='')
(options, args) = parser.parse_args()
prefixList = [prefix.replace("/", "_").replace(":","_") for prefix in options.prefixList.split(',')]
for file_index, prefix in enumerate(prefixList):
    msfile = prefix + '.ms'
    if not os.path.isdir(msfile): importasdm(prefix)
    SPWdic = spwMS(msfile)
    fullResSPWs = GetPHSPWs(msfile)
    if len(fullResSPWs) == 0:
        fullResSPWs = [spw for spw in SPWdic.keys() if SPWdic[spw]['chNum'] >= 8]
    chavSPWs    = GetPHchavSPWs(msfile)
    bandList = list(set([SPWdic[spw]['Band'] for spw in SPWdic.keys()]))
    for band in bandList:
        splitMS = prefix + '.split.%s.ms' % (band)
        os.system('rm -rf %s' % (splitMS))
        bandFullResSPWs = [spw for spw in fullResSPWs if SPWdic[spw]['Band'] == band]
        msmd.open(msfile)
        scanList = msmd.scansforspw(bandFullResSPWs[0]).tolist()
        scanSPWs = msmd.spwsforscan(scanList[0])
        splitSPWs = [int(spw) for spw in scanSPWs if spw in fullResSPWs + chavSPWs]
        msmd.close()
        print('SPLIT : %s %s SPWs=%s' % (prefix, band, str(splitSPWs).strip('[]')))
        split(prefix+'.ms', outputvis=splitMS, spw=str(splitSPWs).strip('[]'),  datacolumn='DATA')
#
