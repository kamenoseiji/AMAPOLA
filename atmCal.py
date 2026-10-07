import sys
import numpy as np
import scipy
from interferometry import GetChNum, GetPSpec, GetLoadTemp, GetTimerecord, get_progressbar_str, indexList, Tcmb, mjd2utc
from casatools import table as tbtool
from casatools import msmetadata as msmdtool
tb = tbtool()
msmd = msmdtool()
#Band      0    1     2     3     4     5     6     7    8    9   10
SBleak = [1.0, 0.01, 0.01, 0.01, 0.03, 0.03, 0.03, 0.1, 0.1, 0.5, 0.5]  # default sideband leakage obtained by au.defaultSBGainsForBand()
SBgain = [0.0, 0.99, 0.99, 0.99, 0.97, 0.97, 0.97, 0.9, 0.9, 0.5, 0.5]  # default sideband gain obtained by au.defaultSBGainsForBand()
#======== Amplitude calibrations
def BlackBody(T, f):    # T in [K], f in [GHz]
    hf_over_k = 4.799243e-2 * f
    return hf_over_k / (np.exp(hf_over_k/T) - 1.0)
#-------- Residuals for Tsky - secz regresion (optical depth + intercept)
def residTskyTransfer( param, Tamb, secz, Tsky, weight ):
    exp_Tau = np.exp( -param[1]* secz )
    return weight* (Tsky - (param[0] + Tcmb* exp_Tau  + Tamb* (1.0 - exp_Tau)))
#
#-------- Residuals for Tsky - secz regresion (without intercept)
def residTskyTransfer0( param, Tamb, secz, Tsky, weight ):
    exp_Tau = np.exp( -param[0]* secz )
    return weight* (Tsky - (Tcmb* exp_Tau  + Tamb* (1.0 - exp_Tau)))
#
#-------- Residuals for Tsky - secz regresion (fixed zenith optical depth)
def residTskyTransfer2( param, Tamb, Tau0, secz, Tsky, weight ):
    exp_Tau = np.exp( -Tau0* secz )
    return weight* (Tsky - (param[0] + Tcmb* exp_Tau  + Tamb* (1.0 - exp_Tau)))
#
def concatScans(timeList, dataList):
    TimeCont, DataCont = [], []
    for scan_index, scan in enumerate(timeList):
        TimeCont += scan.tolist()
        DataCont += dataList[scan_index].tolist()
    index = np.argsort(TimeCont)
    return np.array(TimeCont)[index], np.array(DataCont)[index]
#
def ATTatm(onTime, onData, offTime, offData):   # Attenuator setting between atmCal and onsource
    if np.min(onTime) > np.min(offTime):
        onData = np.array((np.ones(int(min(onTime)) - int(min(offTime)) + 10)* onData[0]).tolist() + onData.tolist())
        onTime = np.array(np.arange(int(min(offTime)) - 10, int(min(onTime))).tolist() + onTime.tolist())
    if np.max(onTime) < np.max(offTime):
        onData = np.array(onData.tolist() + (np.ones(int(max(offTime)) + 10 - int(max(onTime)))* onData[-1]).tolist())
        onTime = np.array(onTime.tolist() + np.arange(int(max(onTime)), int(max(offTime)) + 10 ).tolist())
    smthData = scipy.interpolate.interp1d(onTime, onData, kind='linear')
    return np.median(smthData(offTime) / offData)
#-------- Get atmCal scans
def scanAtmSpec(msfile, useAnt, scanList, spwList, timeOFF=0, timeON=0, timeAMB=0, timeHOT=0):
    timeList, offSpecList, ambSpecList, hotSpecList = [], [], [], []
    antNum, scanNum, spwNum = len(useAnt), len(scanList), len(spwList)
    scanTimeList = []
    scanFlag = list(range(scanNum))
    for scan_index in list(range(scanNum)):
        scanID = scanList[scan_index]
        interval, scanTimeRec = GetTimerecord(msfile, 0, 0, spwList[0], scanID)
        offTime, ambTime, hotTime = np.sort(list(set(scanTimeRec) & set(timeOFF))), np.sort(list(set(scanTimeRec) & set(timeAMB))), np.sort(list(set(scanTimeRec) & set(timeHOT)))
        if (len(offTime)* len(ambTime)* len(hotTime) == 0):
            scanFlag.remove(scan_index)
        else:
            scanTimeList = scanTimeList + [scanTimeRec]
        #
    #
    scanList = np.array(scanList)[scanFlag].tolist()
    scanNum = len(scanTimeList)
    for ant_index in list(range(antNum)):
        progress = (1.0* ant_index + 1.0) / antNum
        sys.stderr.write('\r\033[K' + get_progressbar_str(progress)); sys.stderr.flush()
        for spwID in spwList:
            timeXY, Pspec = GetPSpec(msfile, useAnt[ant_index], spwID)
            chNum = Pspec.shape[1]
            pPol = [0,1]
            if Pspec.shape[0] == 4: pPol = [0,3]
            if Pspec.shape[0] == 1: pPol = [0]
            for scan_index in list(range(scanNum)):
                scanID = scanList[scan_index]
                scanTimeRec = scanTimeList[scan_index]
                offTime, ambTime, hotTime = np.sort(list(set(scanTimeRec) & set(timeOFF))), np.sort(list(set(scanTimeRec) & set(timeAMB))), np.sort(list(set(scanTimeRec) & set(timeHOT)))
                offTimeIndex, ambTimeIndex, hotTimeIndex = indexList(offTime, timeXY),  indexList(ambTime, timeXY),  indexList(hotTime, timeXY)
                if len(offTimeIndex) * len(ambTimeIndex) * len(hotTimeIndex) == 0: continue   # Unuseable atmCal scan
                offTimeIndex, ambTimeIndex, hotTimeIndex = indexList(offTime, timeXY)[-1],  indexList(ambTime, timeXY)[-1],  indexList(hotTime, timeXY)[-1]
                if ((ant_index == 0) & (spwID == spwList[0]) & (len(ambTime) > 0)): timeList = timeList + [offTime[-1]] # Record off-position time
                #
                if ((ant_index == 0) & (spwID == spwList[0]) & (len(ambTime) == 0)):    # No available ambient load data
                    chRange = list(range(int(0.05*chNum), int(0.95*chNum))); chAvgPower = np.mean(Pspec[0][chRange], axis=0)
                    offTimeIndex = indexList(timeOFF, timeXY)
                    ambhotTimeIndex = indexList(timeON, timeXY)
                    ambhotThresh = 0.5*(max(chAvgPower[ambhotTimeIndex]) + min(chAvgPower[ambhotTimeIndex]))
                    hotTimeIndex = np.array(ambhotTimeIndex)[np.where( chAvgPower[ambhotTimeIndex] > ambhotThresh )[0].tolist()].tolist()
                    ambTimeIndex = np.array(ambhotTimeIndex)[np.where( chAvgPower[ambhotTimeIndex] < ambhotThresh )[0].tolist()].tolist()
                    timeList = timeList + [np.median(timeXY[offTimeIndex])]
                #
                if len(ambTime) > 0 :
                    offSpecList = offSpecList + [Pspec[pPol][:,:,offTimeIndex]]
                    ambSpecList = ambSpecList + [Pspec[pPol][:,:,ambTimeIndex]]
                    hotSpecList = hotSpecList + [Pspec[pPol][:,:,hotTimeIndex]]
                    #index += 1
                else:
                    offSpecList = offSpecList + [np.median(Pspec[pPol][:,:,offTimeIndex], axis=2)]
                    ambSpecList = ambSpecList + [np.median(Pspec[pPol][:,:,ambTimeIndex], axis=2)]
                    hotSpecList = hotSpecList + [np.median(Pspec[pPol][:,:,hotTimeIndex], axis=2)]
                #
            #
        #
    #            
    sys.stderr.write('\n'); sys.stderr.flush()
    return np.array(timeList), offSpecList, ambSpecList, hotSpecList, scanList
#
#-------- Trx and Tsky
def TrxTskySpec(useAnt, tempAmb, tempHot, spwList, scanList, ambSpec, hotSpec, offSpec):
    TrxList, TskyList = [], []
    useAntNum, spwNum, scanNum, polNum =  len(useAnt), len(spwList), len(scanList), ambSpec[0].shape[0]
    scanFlag = np.ones([spwNum, polNum, useAntNum, scanNum])
    for spw_index, spw in enumerate(spwList):
        chNum = ambSpec[spw_index* scanNum].shape[1]
        chRange = list(range(int(0.05*chNum), int(0.95*chNum))); chOut = np.sort(list(set(range(chNum)) - set(chRange))).tolist()
        TrxSpec = -np.ones([polNum, chNum, useAntNum, scanNum])
        TskySpec = -np.ones([polNum, chNum, useAntNum, scanNum])
        for pol_index in list(range(polNum)):
            for ant_index in list(range(useAntNum)):
                for scan_index in list(range(scanNum)):
                    index = (ant_index* spwNum + spw_index)* scanNum + scan_index
                    Psamb, Pshot, Psoff = ambSpec[index][pol_index], hotSpec[index][pol_index], offSpec[index][pol_index]
                    if np.max(Pshot[chRange]/Psamb[chRange]) > tempHot[ant_index] / tempAmb[ant_index] : continue       # negative Trx
                    if np.min(Pshot[chRange]/Psamb[chRange]) < 1.01 : continue                                          # infinite Trx
                    TrxSpec[pol_index, chRange, ant_index, scan_index] = (tempHot[ant_index]* Psamb[chRange] - Pshot[chRange]* tempAmb[ant_index]) / (Pshot - Psamb)[chRange]
                    TskySpec[pol_index, chRange, ant_index, scan_index]= (Psoff[chRange]* (tempHot[ant_index] - tempAmb[ant_index]) + tempAmb[ant_index]* Pshot[chRange] - tempHot[ant_index]* Psamb[chRange]) / (Pshot - Psamb)[chRange]
                    TrxSpec[pol_index, chOut, ant_index, scan_index] = np.median(TrxSpec[pol_index, chRange, ant_index, scan_index])       # extraporate band-edge
                    TskySpec[pol_index, chOut, ant_index, scan_index] = np.median(TskySpec[pol_index, chRange, ant_index, scan_index])     # extraporate band-edge
                #
            #
        #
        #-------- Flag negative Trx
        chAvgTrx = np.mean(TrxSpec[:,chRange], axis=1)
        scanFlag[spw_index] = (np.sign(chAvgTrx - 10.0) + 1.0 )/2       # Flag (Trx < 10.0 K) out, scanFlag[spw, pol, ant, scan]
        chAvgTrx = (np.sum(scanFlag[spw_index]* chAvgTrx, axis=2) / np.sum(scanFlag[spw_index] + 1.0e-9, axis=2)).T # chAvgTrx[ant, pol]
        TskySpec = np.sum(TskySpec.transpose(1,0,2,3)* scanFlag[spw_index], axis=1) / np.sum(scanFlag[spw_index] + 1.0e-9, axis=0)  # Average Tsky along polarizations
        TrxList = TrxList + [TrxSpec]
        TskyList = TskyList + [TskySpec]
    #
    return  TrxList, TskyList, scanFlag
#-------- Tsys from ACD
def tau0SpecFit(tempAtm, secZ, useAnt, spwList, TskyList, scanFlag):
    Tau0List, TantNList, Tau0Coef = [], [], []
    scanNum, useAntNum, spwNum = len(secZ), len(useAnt), len(spwList)
    Tau0Excess = np.zeros([spwNum, scanNum])
    #-------- Case1: Single atmCal scan
    if scanNum < 2:
        for spw_index, spw in enumerate(spwList):
            chNum = TskyList[spw_index].shape[0]
            TantNList = TantNList + [np.zeros([useAntNum, chNum])]
            Tau0List  = Tau0List  + [ -np.log( (np.median(TskyList[spw_index], axis=(1,2)) - tempAtm) / (Tcmb - tempAtm) ) / secZ ]
            Tau0Coef = Tau0Coef + [np.zeros(2)]
        return Tau0List, Tau0Excess, Tau0Coef, TantNList
    #    
    #-------- Case2: Multiple atmCal scans, but insuffcient SecZ coverage
    print('SD(secZ) = %f' % (np.std(secZ)))
    if np.std(secZ) < 0.25:
        for spw_index, spw in enumerate(spwList):
            scanWeight = np.sum(scanFlag[spw_index], axis=(0,1))
            chNum = TskyList[spw_index].shape[0]
            TantNList = TantNList + [np.zeros([useAntNum, chNum])]
            Tau0Med = np.zeros(chNum)
            for ch_index in list(range(chNum)):
                param = [0.05]
                #-------- Fit for Tau0 (fixed TantN)
                fit = scipy.optimize.leastsq(residTskyTransfer0, param, args=(tempAtm, secZ, np.nanmedian(TskyList[spw_index][ch_index], axis=0), scanWeight / (np.var(TskyList[spw_index][ch_index], axis=0) + 1e-3) ))
                Tau0Med[ch_index]  = fit[0][0]
            #
            Tau0List  = Tau0List  + [Tau0Med]
            Tau0Excess[spw_index] = residTskyTransfer0([np.median(Tau0Med)], tempAtm, secZ, np.median(TskyList[spw_index], axis=(0,1)), scanWeight ) / (tempAtm - Tcmb)* np.exp(-np.median(Tau0Med)* secZ) / secZ / (scanWeight + 1e-3)
        #
        #-------- Tau0Excess dependent on secZ 
        for spw_index, spw in enumerate(spwList):
            seczSum, seczVar, tauRes = np.sum(secZ),  secZ.dot(secZ), secZ.dot(Tau0Excess[spw_index])
            detTau = scanNum* seczVar - seczSum**2
            coef = np.array([[seczVar, -seczSum],[-seczSum, scanNum]]).dot( np.array([np.sum(Tau0Excess[spw_index]), secZ.dot(Tau0Excess[spw_index])])) / detTau
            Tau0Excess[spw_index] = Tau0Excess[spw_index] - coef[0] - coef[1]* secZ
            Tau0Coef = Tau0Coef + [coef]
        #
        return Tau0List, Tau0Excess, Tau0Coef, TantNList
    #
    #-------- Case3: Suffcient SecZ coverage
    for spw_index, spw in enumerate(spwList):
        param = [0.0, 0.05] # Initial parameter [TantN, Tau0]
        chNum = TskyList[spw_index].shape[0]
        Tau0, TantN = param[1]* np.ones([useAntNum, chNum]), np.zeros([useAntNum, chNum])
        #-------- Fit for Tau0 (without TantN)
        for ant_index in list(range(useAntNum)):
            scanWeight = scanFlag[spw_index, 0, ant_index] * scanFlag[spw_index, 1, ant_index]
            if len(np.where(scanWeight > 0)[0]) > 3:    # at least 4 points to fit
                for ch_index in list(range(chNum)):
                    fit = scipy.optimize.leastsq(residTskyTransfer, param, args=(tempAtm, secZ, TskyList[spw_index][ch_index, ant_index], scanWeight))
                    TantN[ant_index, ch_index] = fit[0][0]
                    Tau0[ant_index, ch_index]  = fit[0][1]
                #
            #
        #
        Tau0Med = np.median(Tau0, axis=0)   # Tau0 is independent on antenna
        #-------- Fit for TantN (fixed Tau0)
        for ant_index in list(range(useAntNum)):
            scanWeight = scanFlag[spw_index, 0, ant_index] * scanFlag[spw_index, 1, ant_index]
            if len(np.where(scanWeight > 0)[0]) > 1:
                for ch_index in list(range(chNum)):
                    param = Tau0[ant_index, ch_index]
                    fit = scipy.optimize.leastsq(residTskyTransfer2, param, args=(tempAtm, Tau0Med[ch_index], secZ, TskyList[spw_index][ch_index, ant_index], scanWeight / (np.var(TskyList[spw_index][ch_index], axis=0) + 1e-3) ))
                    TantN[ant_index, ch_index]  = fit[0][0]
                #
            #
        #
        TskyResid = np.median((TskyList[spw_index].transpose(2,1,0) - TantN), axis=1)
        #-------- Fit for Tau0 (fixed TantN)
        scanWeight = np.sum(scanFlag[spw_index], axis=(0,1))
        for ch_index in list(range(chNum)):
            param = [Tau0Med[ch_index]]
            fit = scipy.optimize.leastsq(residTskyTransfer0, param, args=(tempAtm, secZ, TskyResid[:,ch_index], scanWeight / (np.var(TskyList[spw_index][ch_index], axis=0) + 1e-2)))
            Tau0Med[ch_index]  = fit[0][0]
        #
        Tau0Excess[spw_index] = residTskyTransfer0([np.median(Tau0Med)], tempAtm, secZ, np.median(TskyResid, axis=1), scanWeight ) / (tempAtm - Tcmb)* np.exp(-np.median(Tau0Med)* secZ) / secZ / (scanWeight + 1e-3)
        Tau0List  = Tau0List  + [Tau0Med]
        TantNList = TantNList + [TantN]
        #
    #
    #-------- Tau0Excess dependent on secZ 
    for spw_index in list(range(spwNum)):
        seczSum, seczVar, tauRes = np.sum(secZ),  secZ.dot(secZ), secZ.dot(Tau0Excess[spw_index])
        detTau = scanNum* seczVar - seczSum**2
        coef = np.array([[seczVar, -seczSum],[-seczSum, scanNum]]).dot( np.array([np.sum(Tau0Excess[spw_index]), secZ.dot(Tau0Excess[spw_index])])) / detTau
        Tau0Excess[spw_index] = Tau0Excess[spw_index] - coef[0] - coef[1]* secZ
        Tau0Coef = Tau0Coef + [coef]
        print('SPW%d : coef0=%f coef1=%f' % (spw, coef[0], coef[1]))
    #
    return Tau0List, Tau0Excess, Tau0Coef, TantNList
#
#-------- Log Trx
def LogTrx(antList, spwList, freqList, scanList, timeRef, Trx, TantN, logFile):
    antNum, spwNum, scanNum, polNum = len(antList), len(spwList), Trx[0].shape[3], Trx[0].shape[0]
    for scan_index in list(range(scanNum)):
        text_sd =  'Scan %d : %s' % (scanList[scan_index], mjd2utc(timeRef[scan_index])); logFile.write(text_sd + '\n'); print(text_sd)
        text_sd = 'Trx  : '
        for spw_index in list(range(spwNum)): text_sd = text_sd + ' SPW%03d  %6.1f GHz |' % (spwList[spw_index], freqList[spw_index])
        logFile.write(text_sd + '\n'); print(text_sd)
        text_sd = ' Pol : '
        for spw_index in list(range(spwNum)): text_sd = text_sd + '     X        Y     |'
        logFile.write(text_sd + '\n'); print(text_sd)
        text_sd =  ' ----:-'
        for spw_index in list(range(spwNum)): text_sd = text_sd + '--------------------+'
        logFile.write(text_sd + '\n'); print(text_sd)
        for ant_index in list(range(antNum)):
            text_sd =  antList[ant_index] + ' : '
            for spw_index in list(range(spwNum)):
                for pol_index in list(range(polNum)):
                    text_sd = text_sd + '%7.1f K ' % (np.median(Trx[spw_index], axis=1)[pol_index, ant_index, scan_index])
                text_sd = text_sd + '|'
            logFile.write(text_sd + '\n'); print(text_sd)
        #
    #
    #-------- Log mean and SD in Trx
    text_sd = 'mean : '
    for spw_index in list(range(spwNum)): text_sd = text_sd + '                 SPW%03d  %6.1f GHz |' % (spwList[spw_index], freqList[spw_index])
    logFile.write(text_sd + '\n'); print(text_sd)
    text_sd = ' Pol : '
    for spw_index in list(range(spwNum)): text_sd = text_sd + ' X mean (   sd)    Y mean (   sd)   |'
    logFile.write(text_sd + '\n'); print(text_sd)
    text_sd =  '-----:-'
    for spw_index in list(range(spwNum)): text_sd = text_sd + '------------------------------------+'
    logFile.write(text_sd + '\n'); print(text_sd)
    for ant_index in list(range(antNum)):
        text_sd =  antList[ant_index] + ' : '
        for spw_index in list(range(spwNum)):
            for pol_index in list(range(polNum)):
                text_sd = text_sd + '%7.1f (%5.1f) K ' % (np.median(Trx[spw_index], axis=(1,3))[pol_index, ant_index], np.std(np.median(Trx[spw_index], axis=1)[pol_index, ant_index]) )
            text_sd = text_sd + '|'
        logFile.write(text_sd + '\n'); print(text_sd)
    #
    #-------- Log TantN
    text_sd = 'TantN: '
    for spw_index in list(range(spwNum)): text_sd = text_sd + ' SPW%03d   |' % (spwList[spw_index])
    logFile.write(text_sd + '\n'); print(text_sd)
    text_sd =  ' ----:-'
    for spw_index in list(range(spwNum)): text_sd = text_sd + '----------+'
    logFile.write(text_sd + '\n'); print(text_sd)
    for ant_index in list(range(antNum)):
        text_sd =  antList[ant_index] + ' : '
        for spw_index in list(range(spwNum)):
            text_sd = text_sd + '%7.1f K ' % (np.median(TantN[spw_index][ant_index]))
            text_sd = text_sd + '|'
        logFile.write(text_sd + '\n'); print(text_sd)
    #
    text_sd =  ' ----:-'
    for spw_index in list(range(spwNum)): text_sd = text_sd + '----------+'
    logFile.write(text_sd + '\n'); print(text_sd)
    return
#
