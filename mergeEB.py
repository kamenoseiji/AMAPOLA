import os
import numpy as np
from ASDM_XML import spwMS
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-u', dest='prefixList', metavar='prefixList',
    help='EB UID   e.g. uid___A002_X1399105_X40ca,uid___A002_X1399105_X414b,uid___A002_X1399105_X41c8', default='')
(options, args) = parser.parse_args()
prefixList = [prefix.replace("/", "_").replace(":","_") for prefix in options.prefixList.split(',')]
if 'RB_' in prefixList[0]:
    bandName = 'RB_' + prefixList[0].split('RB_')[1]
else:
    SPWdic = spwMS(prefixList[0] + '.ms')
    bandName = SPWdic[list(SPWdic.keys())[0]]['Band']
prefixElement = prefixList[0].split('_X'); mergeMS = prefixElement[0] + '_X' + prefixElement[1] + '.' + bandName + '.ms'
comvis = []
for file_index, prefix in enumerate(prefixList):
    msfile = prefix + '.ms'
    if os.path.isdir(msfile): comvis = comvis + [msfile]
if len(comvis) > 1:
    print('MERGE : ' + mergeMS)
    os.system('rm -rf %s' % (mergeMS))
    concat(vis=comvis, freqtol='1MHz', dirtol='0.1arcsec', concatvis= mergeMS)
else:
    print('MOVE : ' + mergeMS)
    os.system('mv %s %s' % (comvis[0], mergeMS))
