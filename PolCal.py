import os
import numpy as np
from interferometry import ALMA_lat, GetAzEl, AzElMatch, AzEl2PA
from casatools import msmetadata as msmdtool
msmd = msmdtool()
#
#-------- Get Stokes Parameters from AMAPOLA
def GetAMAPOLAStokes(SCR_DIR, sourceList, timeText, FreqGHz):    # 
    # sourceList : e.g. ['J1256-0547', 'J1924-2914']
    # timeText   : e.g. '2017/04/12/11:26:17'
    # FreqGHz    : frequency in GHz
    StokesDic = dict(zip(sourceList, [[]]*len(sourceList)))   # Stokes parameters for each source
    os.system('rm -rf CalQU.data')
    text_sd = 'Rscript %spolQuery.R -D%s -F%f' % (SCR_DIR, timeText, FreqGHz)
    for source in sourceList: text_sd = text_sd + ' ' + source
    print(text_sd)
    os.system(text_sd)
    fp = open('CalQU.data')
    lines = fp.readlines()
    fp.close()
    for eachLine in lines:
        sourceName = eachLine.split()[0]
        I, Q, U = float(eachLine.split()[1]), float(eachLine.split()[2]), float(eachLine.split()[3])
        if abs(Q) > 0.5*I: Q = 0.0
        if abs(U) > 0.5*I: U = 0.0
        StokesDic[sourceName] = [FreqGHz, I, Q, U, 0.0]
    #
    return StokesDic
#
#-------- PA and polarization responses
def PolResponse(msfile, srcDic, StokesDic, BandPA, scanList, mjdList):
    scanDic = dict(zip(scanList, [[]]* len(scanList)))
    azelTime, AntID, AZ, EL = GetAzEl(msfile)
    msmd.open(msfile)
    #-------- Check AZEL
    print('------------------- AMAPOLA-based prediction -----------------')
    print('        Source     :    I     p%     EVPA  QCpUS  UCmQS   EL  ')
    print('-------+-----------+-------+------+------+------+------+------')
    for scan_index, scan in enumerate(scanList):
        fieldID = msmd.fieldsforscan(scan)[0]
        sourceName = srcDic[fieldID]['Name']
        AzScan, ElScan = AzElMatch(mjdList[scan_index], azelTime, AntID, AZ, EL)
        PA = AzEl2PA(AzScan, ElScan) + BandPA
        CS, SN, QCpUS, UCmQS = np.cos(2.0* PA), np.sin(2.0* PA), np.zeros(len(PA)), np.zeros(len(PA))
        StokesI, QCpUS, UCmQS = 0.0, 0.0, 0.0
        if str.isdigit(sourceName[1]) and len(StokesDic[sourceName]) > 0:
            StokesI = StokesDic[sourceName][1]
            QCpUS = StokesDic[sourceName][2]*CS + StokesDic[sourceName][3]*SN   # Qcos + Usin
            UCmQS = StokesDic[sourceName][3]*CS - StokesDic[sourceName][2]*SN   # Ucos - Qsin
            BPquality = StokesDic[sourceName][1]* np.sin(np.median(ElScan) - 0.5)  # cut at 40 deg
            XYquality = np.median(abs(QCpUS))* np.sin(np.median(ElScan) - 0.5)     # cut at 30 deg
            print('Scan%3d %s : %6.2f %6.2f %6.2f %6.2f %6.2f %6.2f' % (scan, sourceName, StokesDic[sourceName][1], 100.0*np.sqrt(StokesDic[sourceName][2]**2 + StokesDic[sourceName][3]**2)/StokesDic[sourceName][1], 90.0* np.arctan2(StokesDic[sourceName][3], StokesDic[sourceName][2])/np.pi, np.median(QCpUS), np.median(UCmQS), 180.0* np.median(ElScan)/np.pi))
        #
        scanDic[scan] = {
            'msfile': msfile,
            'source': sourceName,
            'mjdSec': mjdList[scan_index],
            'AZ'    : AzScan,
            'EL'    : ElScan,
            'PA'    : PA,
            'SA'    : srcDic[fieldID]['SA'],
            'I'     : StokesI,
            'QCpUS' : QCpUS,
            'UCmQS' : UCmQS}
    msmd.close(); msmd.done()
    return scanDic
#
