import numpy as np
from Plotters import plotTsysDic
from interferometry import GetAntName, GetChNum, GetBandNames, GetTimerecord, GetAzEl, GetTemp, GetPSpecScan, GetPSpec, RADDEG, Tcmb, mjd2utc, indexList, get_progressbar_str
from atmCal import ATTatm
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-u', dest='prefix', metavar='prefix',
    help='EB UID   e.g. uid___A002_X10dadb6_X18e6', default='')
parser.add_option('-t', dest='PLOTTSYS', metavar='PLOTTSYS',
    help='Plot Tsys and Trx spectra', action="store_true")
(options, args) = parser.parse_args()
prefix  = options.prefix.replace("/", "_").replace(":","_").replace(" ","")
PLOTTSYS= options.PLOTTSYS
'''
prefix = 'uid___A002_X102faf1_Xe5d'
PLOTTSYS = True
'''
#-------- Read SYSCAL table
msfile = prefix + '.ms'
tb.open(msfile + '/SYSCAL')
colnameList = tb.colnames()
timeStamp = tb.getcol('TIME')
tempAtm = GetTemp(msfile)
#timeSyscalList = timeStamp[ np.where(np.diff(timeStamp) != 0.0)[0].tolist() + [-1] ]
timeSyscalList = np.unique(timeStamp)
antID = tb.getcol('ANTENNA_ID')
spwID = tb.getcol('SPECTRAL_WINDOW_ID')
TRX = tb.getcol('TRX_SPECTRUM')
TSY = tb.getcol('TSYS_SPECTRUM')
tb.close()
antList, spwList = GetAntName(prefix + '.ms'), np.unique(spwID)
azelTime, AntID, AZ, EL = GetAzEl(msfile)
refAntID = np.unique(AntID)[0]
azelRefIndex = np.where(AntID == refAntID)[0].tolist()
azelTime, AZ, EL = azelTime[azelRefIndex], AZ[azelRefIndex], EL[azelRefIndex]
spwDic = dict(zip(spwList, [[]]*len(spwList)))  # SPW
antNum, spwNum = len(antList), len(spwList)
msmd.open(msfile)
scanList = msmd.scansforintent("CALIBRATE_ATMOSPHERE*").tolist()
timeOFF  = msmd.timesforintent("CALIBRATE_ATMOSPHERE#OFF_SOURCE")
timeON   = msmd.timesforintent("CALIBRATE_PHASE#ON_SOURCE")
timeList = [np.median(msmd.timesforscan(scan)) for scan in scanList]
chavspwList = list((set(msmd.almaspws(chavg=True)) - set(msmd.almaspws(sqld=True))) & set(msmd.spwsforscan(scanList[0])))
OnScanList  = list((set(msmd.scansforintent('*ON_SOURCE')) - set(msmd.scansforintent('*ATMOSPHERE*'))) & set(msmd.scansforspw(chavspwList[0])))
onTimeList = [np.median(msmd.timesforscan(scan)) for scan in OnScanList]
msmd.close()
OnScanList.sort()
BandNames = GetBandNames(msfile, spwList); UniqBands = list(set(BandNames))
NumBands = len(UniqBands)
TsysDic = dict(zip(scanList, [[]]*len(scanList)))
#-------- Read SPW frequency
for spw_index, spw in enumerate(spwList):
    chNum, chWid, freq = GetChNum(msfile, spw)
    spwDic[spw] = {'band': BandNames[spw_index], 'chNum': chNum, 'chWid': chWid, 'freq' : freq}
for scan_index, scan in enumerate(scanList):
    if scan_index >= len(timeSyscalList):
        scanList.remove(scan)
        del TsysDic[scan]
        continue
    data_index = np.where(timeStamp == timeSyscalList[scan_index])[0].tolist()
    spwsInScan = np.unique(spwID[data_index]).tolist()
    antsInScan = np.unique(antID[data_index]).tolist()
    timeLabel = 'Scan %d : %s' % (scan, mjd2utc(timeList[scan_index])[:-4])
    azelRefIndex = np.argmin(abs(azelTime - timeList[scan_index]))
    TsysDic[scan] = {
        'Band'   : spwDic[spwsInScan[0]]['band'],
        'antList': antList[antsInScan],
        'mjdSec' : timeList[scan_index],
        'Az'     : AZ[azelRefIndex],
        'El'     : EL[azelRefIndex],
        'SecZ'   : 1.0/np.sin(EL[azelRefIndex]),
        'spwList': spwsInScan,
        'chNum'  : [spwDic[spw]['chNum'] for spw in spwsInScan],
        'freq'   : [spwDic[spw]['freq']*1.0e-9 for spw in spwsInScan],
        'Trx'    : TRX[:,:,data_index],
        'Tsys'   : TSY[:,:,data_index],
        'Tsky'   : np.mean((TSY[:,:,data_index] - TRX[:,:,data_index])* tempAtm / (TSY[:,:,data_index] + tempAtm), axis=0)} 
#-------- Save Trx and Tsky 
for bandName in np.unique(BandNames):
    logFile  = open('%s-%s-TelCal.log' % (prefix, bandName), 'w')
    bandScanList = [scan for scan in TsysDic.keys() if TsysDic[scan]['Band'] == bandName]
    TsysBandDic = {keyScan: dicValue for keyScan, dicValue in TsysDic.items() if keyScan in bandScanList}
    #-------- Save into NPY
    spwNum = len(TsysBandDic[bandScanList[0]]['spwList'])
    np.save('%s-%s.TrxAnt.npy' % (prefix, bandName), antList)     # antList[ant]
    for spw_index, spw in enumerate(TsysBandDic[bandScanList[0]]['spwList']):
        np.save('%s-%s-SPW%d.TrxFreq.npy' % (prefix, bandName, spw), TsysBandDic[bandScanList[0]]['freq'][spw_index])
        TrxList, TskyList = [], []
        for scan_index, scan in enumerate(TsysBandDic.keys()):
            Trx  = TsysBandDic[scan]['Trx'][:,:,range(spw_index, antNum*spwNum, spwNum)]
            Tsky = TsysBandDic[scan]['Tsky'][:,range(spw_index, antNum*spwNum, spwNum)]
            TrxList  = TrxList  + [Trx]
            TskyList = TskyList + [Tsky]
        np.save('%s-%s-SPW%d.Trx.npy' % (prefix, bandName, spw), np.array(TrxList).transpose(1,2,3,0))
        np.save('%s-%s-SPW%d.Tsky.npy' % (prefix, bandName, spw), np.array(TskyList).transpose(1,2,0))
    #-------- Log Trx 
    atmTimeList, atmElList = [], []
    for scan_index, scan in enumerate(scanList):
        atmTimeList = atmTimeList + [TsysBandDic[scan]['mjdSec']]
        atmElList   = atmElList + [TsysBandDic[scan]['El']]
        timeLabel = 'Scan %d : %s EL=%.1f' % (scan, mjd2utc(TsysBandDic[scan]['mjdSec'])[:-4], RADDEG*TsysBandDic[scan]['El'])
        logFile.write(timeLabel + '\n'); print(timeLabel)
        text_sd = 'Trx  : '
        for spw_index, spw in enumerate(TsysBandDic[scan]['spwList']):
            text_sd = text_sd + ' SPW%03d  %6.1f GHz |' % (spw, np.median(TsysBandDic[scan]['freq'][spw_index]))
        logFile.write(text_sd + '\n'); print(text_sd)
        text_sd = ' Pol : '
        for spw_index, spw in enumerate(TsysBandDic[scan]['spwList']): text_sd = text_sd + '     X         Y    |'
        logFile.write(text_sd + '\n'); print(text_sd)
        text_sd =  ' ----:-'
        for spw_index, spw in enumerate(TsysBandDic[scan]['spwList']): text_sd = text_sd + '--------------------+'
        logFile.write(text_sd + '\n'); print(text_sd)
        for ant_index, ant in enumerate(TsysBandDic[scan]['antList']):
            text_sd = '%s : ' % (ant)
            for spw_index, spw in enumerate(TsysBandDic[scan]['spwList']):
                data_index = spwNum* ant_index + spw_index
                for pol_index in [0,1]: text_sd = text_sd + '%7.1f K ' % (np.median(TsysBandDic[scan]['Trx'][pol_index][:,data_index], axis=0))
                text_sd = text_sd + '|'
            logFile.write(text_sd + '\n'); print(text_sd)
    np.save('%s-%s.atmTime.npy' % (prefix, bandName), np.array(atmTimeList))
    np.save('%s-%s.EL.npy' % (prefix, bandName), np.array(atmElList))
    logFile.close()
    if PLOTTSYS: plotTsysDic(prefix, TsysBandDic)
    if len(scanList) > 1: continue
    #-------- fill Tsys records in on-source scans
    atmEl = np.ones(len(OnScanList))
    for scan_index, onTime in enumerate(onTimeList):
        atmEl[scan_index] = EL[np.argmin(abs(azelTime - onTime))]
    np.save('%s-%s.atmTime.npy' % (prefix, bandName), np.array(onTimeList))
    np.save('%s-%s.EL.npy' % (prefix, bandName), atmEl)
    chavspwList.sort()
    #-------- CHAV data
    for spw_index, spw in enumerate(chavspwList):
        Tsky = np.median(TsysDic[scanList[0]]['Tsky'][:,range(spw_index, antNum*spwNum, spwNum)], axis=1)
        TskyScan = np.ones([chNum, antNum, len(OnScanList)])
        TrxScan  = np.ones([2, chNum, antNum, len(OnScanList)])
        for ant_index, ant in enumerate(antList):
            #-------- CHAV in the atmcal scan
            timeScan, CHAV = GetPSpecScan(msfile, ant_index, spw, scanList[0])
            offIndex = indexList(timeOFF, timeScan)
            offCHAVCont = np.mean(CHAV[:,0,offIndex], axis=1)
            offCHAVCont = offCHAVCont[::len(offCHAVCont)-1] 
            Trx = TsysBandDic[scanList[0]]['Trx'][:,:,range(spw_index, antNum*spwNum, spwNum)[ant_index]] # Trx[pol, ch]
            GainAtm = offCHAVCont / (Trx + Tsky).T     # GainAtm[ch, pol]
            #-------- CHAV in the first on-source scan
            timeScan, CHAV = GetPSpecScan(msfile, ant_index, spw, OnScanList[0])
            CHAVCont = np.mean(CHAV[:,0], axis=1)
            CHAVCont = CHAVCont[::len(CHAVCont)-1] 
            scaleFact = CHAVCont / offCHAVCont
            #-------- Tsky for all on-source scans
            timeScan, CHAV = GetPSpec(msfile, ant_index, spw)
            for scan_index, scan in enumerate(OnScanList):
                progress = ((spw_index* antNum + ant_index)* len(OnScanList) + scan_index + 1.0) / (len(chavspwList)* antNum* len(OnScanList))
                sys.stderr.write('\r\033[K' + get_progressbar_str(progress)); sys.stderr.flush()
                CHAVCont = np.mean(CHAV[:, 0, np.where( abs( timeScan - onTimeList[scan_index]) < 5.0 )[0].tolist()], axis=1)   
                CHAVCont = CHAVCont[::len(CHAVCont)-1] / scaleFact 
                TrxScan[:,:,ant_index,scan_index]= Trx
                TskyScan[:,ant_index,scan_index] = np.mean((CHAVCont / GainAtm).T - Trx, axis=0)
            # end scan loop
        # end antenna loop
        np.save('%s-%s-SPW%d.Trx.npy' % (prefix, bandName, TsysDic[scanList[0]]['spwList'][spw_index]), TrxScan)
        np.save('%s-%s-SPW%d.Tsky.npy' % (prefix, bandName, TsysDic[scanList[0]]['spwList'][spw_index]), TskyScan)
    # end spw loop
    print('')
#
