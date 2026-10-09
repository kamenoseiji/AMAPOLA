import os
import glob
import numpy as np
from datetime import datetime
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-u', dest='prefixList', metavar='prefixList',
    help='EB UID   e.g. uid___A002_X13e5a87_X34eb,uid___A002_X13e5a87_X3570,uid___A002_X13e5a87_X35ed', default='')
(options, args) = parser.parse_args()
prefixList = [prefix for prefix in options.prefixList.split(',')]
for prefix in prefixList: os.system('rm -rf %s-*-Flux.log' % (prefix))
prefixElement = prefixList[0].split('_X'); newPrefix = prefixElement[0] + '_X' + prefixElement[1]
logFileList = glob.glob(newPrefix + '*Flux.log')
for logFileName in logFileList:
    band = logFileName.split('.')[1].split('-')[0]
    logfile = open(logFileName, 'r')
    logLines = logfile.readlines()
    logfile.close()
    #-------- Read each lines
    commonHeader, commonAe, commonDterm, ScanLog, DT = ['# Combined with'], [], [], [], []
    for prefix in prefixList: commonHeader[0] = commonHeader[0] + ' ' + prefix
    commonHeader[0] = commonHeader[0] + '\n'
    AeSection    = False
    DtermSection = False
    ScanSection  = -1
    for Line_index, Line in enumerate(logLines):
        if Line[0] == '#' :
            commonHeader = commonHeader + [Line]
            AeSection    = False
            DtermSection = False
            ScanSection  = -1
            continue
        if Line.split(':')[0] == ' Aeff' :
            commonAe = commonAe + [Line]
            AeSection    = True
            DtermSection = False
            ScanSection  = -1
            continue
        if Line.split()[0] == 'D-term' :
            commonDterm = commonDterm + [Line]
            AeSection    = False
            DtermSection = True
            ScanSection  = -1
            continue
        if str.isdigit(Line.split()[0]):
            AeSection    = False
            DtermSection = False
            ScanSection += 1
            DT = DT + [datetime.strptime(Line.split()[4], "%Y/%m/%d/%H:%M:%S").timestamp()]
        if AeSection:    commonAe = commonAe + [Line]
        if DtermSection: commonDterm = commonDterm + [Line]
        if ScanSection >= 0: ScanLog = ScanLog + ['SCAN %d #' % (ScanSection) + Line]
    #-------- Separate into each EB
    DTdiff = np.diff(DT)
    DTgap = float(np.sort(DTdiff)[::-1][len(prefixList)-2])
    scanDelimitor = [diff_index for diff_index,diff in enumerate(DTdiff) if diff >= DTgap]
    scanDelimitor = scanDelimitor + [len(DT)-1]
    scanInitilize = [0] + [scan + 1 for scan in scanDelimitor[:-1]]
    for file_index, prefix in enumerate(prefixList):
        fileName = prefix + '-' + band + '-Flux.log'
        splitlogfile = open(fileName, 'w')
        for logEntry in commonHeader: splitlogfile.write(logEntry)
        for logEntry in commonAe: splitlogfile.write(logEntry)
        for logEntry in ScanLog:
            ScanID = int(logEntry.split()[1])
            log2write = logEntry.split('#')[1]
            if (ScanID < scanInitilize[file_index]) or (ScanID > scanDelimitor[file_index]): continue
            splitlogfile.write(log2write)
        for logEntry in commonDterm: splitlogfile.write(logEntry)
        splitlogfile.close()
#
