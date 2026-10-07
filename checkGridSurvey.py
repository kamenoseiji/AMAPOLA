import os
import glob
import re
import math
import numpy as np
import xml.etree.ElementTree as ET
from matplotlib.backends.backend_pdf import PdfPages
from interferometry import GetTemp, Tcmb, kb, BANDPA, BANDFQ, indexList, subArrayIndex, GetAntName, GetAntD, GetSourceDic, GetBandNames, GetAeff, GetDterm, quadratic_interpol, GetAtmSPWs, GetBPcalSPWs, GetSPWFreq, GetOnSource, GetUVW, loadScanSPW, gainComplex, gainComplexErr, gainComplexVec, bestRefant, ANT0, ANT1, Ant2Bl, Ant2BlD, Bl2Ant, CrossPolBL, CrossPolBP, SPWalign, delay_search, linearRegression, VisMuiti_solveD, AllanVarPhase, specBunch
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
from Plotters import plotSP, plotXYP, plotBLAV, plotFL, plotQUXY, plotResidualMap
from Grid import *
from ASDM_XML import CheckCorr, BandList
from PolCal import GetAMAPOLAStokes, PolResponse
from optparse import OptionParser
def AV(vis): return AllanVarPhase(np.angle(vis), 1)
def GetSSOFlux(StokesDic, timeText, FreqGHz):
    from Grid import SSOCatalog
    # StokesDic  : Stokes parameter dictionlary
    # timeText   : e.g. '2017/04/12/11:26:17'
    # FreqGHz    : frequency list in GHz
    #sourceList = list(StokesDic.keys())
    #SSOList = [source for source in sourceList if not str.isdigit(source[1])]
    SSOList = [source for source in StokesDic.keys() if source in SSOCatalog]
    SSODic = dict(zip(SSOList, [[]]*len(SSOList)))
    if len(SSOList) == 0: return StokesDic, SSODic
    for SSO in SSOList:
        flux, major, minor, pa = [], [], [], []
        for freq in FreqGHz:
            if freq >= 62.0:
                text_Freq = '%.2fGHz' % (freq)
                SSOmodel = predictcomp(objname=SSO, standard="Butler-JPL-Horizons 2012", minfreq=text_Freq, maxfreq=text_Freq, nfreqs=1, prefix="", antennalist="aca.cycle3.cfg", epoch=timeText, showplot=True)
                flux  = flux + [SSOmodel['spectrum']['bl0flux']['value']]
            else:
                text_Freq = '62.0GHz'
                SSOmodel = predictcomp(objname=SSO, standard="Butler-JPL-Horizons 2012", minfreq=text_Freq, maxfreq=text_Freq, nfreqs=1, prefix="", antennalist="aca.cycle3.cfg", epoch=timeText, showplot=True)
                flux  = flux + [SSOmodel['spectrum']['bl0flux']['value']* (freq/62.0)**2]
        major = SSOmodel['shape']['majoraxis']['value']* np.pi / 21600.0
        minor = SSOmodel['shape']['minoraxis']['value']* np.pi / 21600.0
        pa    = SSOmodel['shape']['positionangle']['value']* np.pi / 180.0
        StokesDic[SSO] = [FreqGHz, flux]
        SSODic[SSO]    = [FreqGHz, flux, [major, minor, pa]]
    return StokesDic, SSODic
SCR_DIR = os.environ['HOME'] + '/ALMA_Py3/'
TBL_DIR = 'https://www.alma.cl/~skameno/AMAPOLA/Table/'
#-------- Parse options
parser = OptionParser()
parser.add_option('-u',  dest='uid', metavar='uid', help='UID to reduce e.g. uid___A002_X10ded83_Xa91e', default='')
parser.add_option('-a',  dest='antFlag', metavar='antFlag', help='Antennas to flag out', default='')
parser.add_option('-A',  dest='Apriori', metavar='Apriori',  help='Apriori amplitude calibration', action='store_true')
parser.add_option('-Q',  dest='QUMODEL', metavar='QUMODEL', help='Initial Q,U from AMAPOLA', action='store_true')
parser.add_option('-U',  dest='uvLimit', metavar='uvLimit', help='(u, v) limet [m]', default='5000')
parser.add_option('-c',  dest='Scan',    metavar='Scan',    help='Scan number to refer', default='0')
parser.add_option('-X',  dest='XYsign',  metavar='XYsign',  help='Force XY sign (1 or -1)', default='')
parser.add_option('-m',  dest='Map',     metavar='Map',     help='Save redisual map', action='store_true')
parser.add_option('-R', dest='refant', metavar='refant',    help='reference antenna', default='')
parser.add_option('-T',  dest='TsysDigital', metavar='TsysDigital',  help='Apply Tsys digital correction', action='store_true')
(options, args) = parser.parse_args()
#-------- Check options
prefix  = options.uid.replace("/", "_").replace(":","_").replace(" ","")
antFlag = [ant for ant in options.antFlag.split(',') if ant != '']
uvLimit = int(options.uvLimit) 
refant  = options.refant
BPscan  = int(options.Scan)
QUMODEL = options.QUMODEL
Apriori = options.Apriori
PLOTMAP = options.Map
PLOTFL  = PLOTMAP
PLOTQU  = PLOTMAP
TsysDigitalCorrection = options.TsysDigital
if options.XYsign != '': givenXYsign  = np.sign(int(options.XYsign))
'''
prefix = 'uid___A002_X13e5a87_X34eb'
antFlag = []
uvLimit = 5000
refant  = ''
BPscan  = 0
QUMODEL = True
TsysDigitalCorrection = False
PLOTMAP = False
PLOTFL = False
PLOTQU = False
'''
msfile = prefix + '.ms'
tempAtm = GetTemp(msfile)
if tempAtm != tempAtm: tempAtm = 270.0; print('Cannot get ambient-load temperature ... employ 270.0 K, instead.')
#-------- Tsys calibration
os.system('casa --nologger --agg -c ' + SCR_DIR + 'SysCalTrx.py -u %s' % (prefix))
#os.system('casa --nologger --agg -c ' + SCR_DIR + 'TsysCal.py -u %s' % (prefix))
TrxFileList = glob.glob(prefix + '*Trx.npy'); TrxFileList.sort()
RXList = list(set( [TrxFile.split('-')[1] for TrxFile in TrxFileList] ))
BandName = RXList[0]
#-------- Check Correlator Type
BLCORR = True
if 'ACA' in CheckCorr(prefix): BLCORR = False
#-------- Check Antenna List
antList = GetAntName(msfile)
antFlag = [ant for ant in antFlag if ant in antList]
antDia = GetAntD(antList)
antDic = ArrayAltAz(msfile) # Antenna position in (E,N,H) coordinates, relative to the array center
antNum = len(antList)
blNum = int(antNum* (antNum - 1) / 2)
polXindex, polYindex = (np.arange(4)//2).tolist(), (np.arange(4)%2).tolist()
#-------- Check SPWs of atmCal and bandpass
print('---Checking SPWs and Scan information')
bpSPWs = GetBPcalSPWs(msfile)
OnScanList = GetOnSource(msfile)
if len(OnScanList) == 0: RXList = []
#-------- 
msmd.open(msfile)
spwNames = msmd.namesforspws(bpSPWs)
if BandName not in ['RB_01', 'RB_02', 'RB_03', 'RB_04', 'RB_06', 'RB_07']: msmd.close(); exit()
#-------- Prepare log files
logfile = open(prefix + '-' + BandName + '-Flux.log', 'w')
text_corr = ' %s BLCORR' % (prefix) if BLCORR else ' %s ACACORR' % (prefix)
text_tsysdigital = ' TsysDigitalCorrection ON' if TsysDigitalCorrection else ' TsysDigitalCorrection OFF'
logfile.write('# ' + text_corr + text_tsysdigital + '\n')
#-------- Frequency setup
BandPA = (BANDPA[int(BandName[3:5])] + 90.0)*math.pi/180.0
BandbpSPW = {'spw': [spw for spw_index,spw in enumerate(bpSPWs) if BandName in spwNames[spw_index]]}
BandatmSPW= {'spw': [int(TrxFile.split('SPW')[1].split('.')[0]) for TrxFile in TrxFileList if BandName in TrxFile] }
BandatmSPW['spw'].sort()
BandScanList = list(set(msmd.scansforspw(BandbpSPW['spw'][0])) & set(OnScanList))
BandScanList.sort()
BandbpSPW = GetSPWFreq(msfile, BandbpSPW)   # BandbpSPW : [[SPW List][freqArray][chNum][BW]]
BandatmSPW = GetSPWFreq(msfile, BandatmSPW)
#---- Identify check scan (scan to check Allan Variance)
if BPscan == 0:     # BPscan was not specified
    checkScan = list(set(msmd.scansforintent('*BANDPASS*')) & set(BandScanList))
    if len(checkScan) == 0: checkScan = list(set(msmd.scansforintent('*FLUX*')) & set(BandScanList))
    if len(checkScan) == 0: checkScan = list(set(msmd.scansforintent('*PHASE*')) & set(BandScanList))
    if len(checkScan) == 0: checkScan = list(set(msmd.scansforintent('*AMPLI*')) & set(BandScanList))
    if len(checkScan) == 0: checkScan = list(set(msmd.scansforintent('*DELAY*')) & set(BandScanList))
    if len(checkScan) == 0: checkScan = list(set(msmd.scansforintent('*POINTING*')) & set(BandScanList))
    checkScan.sort()
    checkScan = checkScan[0]
else:
    checkScan = BPscan
print('---Checking usable antennas for %s by ASD in Scan %d' % (BandName, checkScan))
chavSPWs = list((set(msmd.chanavgspws()) - set(msmd.almaspws(sqld=True)) - set(msmd.almaspws(wvr=True))) & set(msmd.spwsforscan(checkScan))); chavSPWs.sort()
timeStampList, XspecList = loadScanSPW(msfile, chavSPWs, [checkScan])  # XspecList[spw][scan] [corr, ch, bl, time]
#---- Check Allan Variance of channel-averaged visibilities in all baselines in the check scan
for spw_index, spw in enumerate(chavSPWs):
    Trx = np.load('%s-%s-SPW%d.Trx.npy'  % (prefix, BandName, BandatmSPW['spw'][spw_index]))
    antFlag = antFlag + antList[np.where(np.min(np.median(Trx, axis=(1,3)), axis=0) < 5)[0].tolist()].tolist()  # flag unrealistic Trx out
    antFlag = antFlag + antList[np.where(np.max(np.median(Trx, axis=(1,3)), axis=0) > 2.0* np.median(Trx))[0].tolist()].tolist()  # flag unrealistic Trx out
    checkVis = XspecList[spw_index][0][[0,-1]][:,0]
    AV_bl = np.array([np.apply_along_axis(AV, 1, checkVis[0]), np.apply_along_axis(AV, 1, checkVis[1])])
    plotBLAV(prefix, antList, spw, AV_bl)
    AV_bl = np.sum(AV_bl, axis=0)* antDia[ANT0[0:blNum]]*antDia[ANT1[0:blNum]]
    antFlag = list(set(antFlag))
    #-------- Antennas in bad baselines
    for badAnt in antFlag:
        badAntIndex = np.where(antList==badAnt)[0][0]
        badBL = [Ant2Bl(badAntIndex, index) for index in list(range(antNum)) if index != badAntIndex]
        AV_bl[badBL] = np.median(AV_bl) 
    errBL = np.where(AV_bl > 25.0* np.median(AV_bl))[0].tolist()
    #-------- Identify bad antennas from bad baselines
    while len(errBL) > 0:
        flagAntCount = np.unique([Bl2Ant(bl) for bl in errBL], return_counts=True) 
        badAnt = antList[flagAntCount[0][np.argmax(flagAntCount[1])]]
        antFlag = list(set(antFlag + [badAnt]))
        for badAnt in antFlag:
            badAntIndex = np.where(antList==badAnt)[0][0]
            badBL = [Ant2Bl(badAntIndex, index) for index in list(range(antNum)) if index != badAntIndex]
            AV_bl[badBL] = np.median(AV_bl)
        errBL = np.where( AV_bl > 25.0* np.median(AV_bl) )[0].tolist()
antFlag =list(set(antFlag))
text_sd = '---Flagged by Trx and Allan Variance (%d): ' % (len(antFlag))
for ant in antFlag: text_sd = text_sd + '%s ' % (ant)
print(text_sd); logfile.write('#' + text_sd + '\n')
flagAnt = np.array([0.0 if ant in antFlag else 1.0 for ant in antList])
flagBL = flagAnt[ANT0[0:blNum]]* flagAnt[ANT1[0:blNum]]
useAntMap = np.where(flagAnt > 0.1)[0].tolist()
useBlMap = np.where(flagBL > 0.1)[0].tolist()
#---- A priori antenna paremeters (D-terms, aperture)
Dcat = GetDterm(TBL_DIR, antList, int(BandName[3:5]), timeStampList[0][0])          # D-term catlogue
etaA = 0.01* GetAeff(TBL_DIR, antList, int(BandName[3:5]), timeStampList[0][0]).T   # Aperture efficiency catalogue
#---- Source List : identify solar system objects and quasars with a priori Stokes parameters
print('-----Estimation from AMAPOLA and Butler-JPL-Horizons')
srcDic = GetSourceDic(msfile)
sourceList = list(dict.fromkeys([ srcDic[ID]['Name'] for ID in srcDic.keys() ]))
SSOList   = indexList( np.array(SSOCatalog), np.array(sourceList))
FscaleDic = dict(zip(np.array(sourceList)[SSOList].tolist(), [[]]* len(SSOList)))   # Dictionary of solar system objects for flux calibration
StokesDic = GetAMAPOLAStokes(SCR_DIR, sourceList, qa.time('%fs' % (timeStampList[0][0]), form='ymd')[0], BANDFQ[int(BandName[3:5])])    # QSO: [FreqGHz, IJy, QJy, UJy, VJy]
StokesDic, SSODic = GetSSOFlux(StokesDic, qa.time('%fs' % (timeStampList[0][0]), form='ymd')[0], [1.0e-9* np.median(BandbpSPW['freq'][spw_index]) for spw_index, spw in enumerate(BandbpSPW['spw'])])   # SSODic[SSO] : [[frequencies for spw], [flux for spw], [maxsize(rad), minsize(rad), pa]]
#-------- Load Visibilities into memory
timeStampList, XspecList = loadScanSPW(msfile, BandbpSPW['spw'], BandScanList)  # XspecList[spw][scan] [corr, ch, bl, time]
#-------- Polarization responses per scan
scanDic = PolResponse(msfile, srcDic, StokesDic, BandPA, BandScanList, timeStampList)
#-------- Identity shadowed antennas
for scan in scanDic.keys(): 
    antDic = AntShadow(antDic, np.median(scanDic[scan]['AZ']), np.median(scanDic[scan]['EL']))
    scanDic[scan]['Shadow'] = [ant for ant in antDic.keys() if antDic[ant]['Shadow'] > 0.05]
QSOscanList = [scan for scan in scanDic.keys() if scanDic[scan]['I'] > 0.01 ]
#-------- Apply Tsys calibration
scanDic, XspecList = applyTsysCal(prefix, BandName, BandbpSPW, scanDic, SSODic, XspecList)
#-------- Select reference polcal scan (max XY cross correlation)
if BPscan == 0:
    polCalScan = QSOscanList[np.argmax(np.array([np.median(abs(scanDic[scan]['UCmQS']))* scanDic[scan]['I']* np.sign(np.median(scanDic[scan]['EL']) - ELshadow) for scan in QSOscanList]))]
else:
    polCalScan = BPscan
polCalSource = scanDic[polCalScan]['source']
text_sd = '-----PolCal Scan %d : %s' % (polCalScan, polCalSource)
print(text_sd); logfile.write('#' + text_sd + '\n')
#-------- Check coherence and aperture effieincy in check scan
for spw_index, spw in enumerate(BandbpSPW['spw']):
    Xspec = XspecList[spw_index][BandScanList.index(polCalScan)][:,:,useBlMap]          # Xspec[pol,ch,bl,time]
    checkVis    = np.mean(Xspec[[0,-1]][:,BandbpSPW['chRange'][spw_index]], axis=1) / scanDic[polCalScan]['I']  # checkVis[pol, bl, time]
    Gain =  np.array([np.apply_along_axis(gainComplex, 0, checkVis[0]), np.apply_along_axis(gainComplex, 0, checkVis[-1])]) # Gain[pol, ant, time]
    antCoh = np.array([np.median(abs(Gain[0,ant_index]* Gain[1,ant_index].conjugate())) for ant_index, ant in enumerate(useAntMap)])    # antCoh[ant]
    Aeff = 8.0* kb* antCoh / (np.pi* antDia[useAntMap]**2)  # Aeff[ant] : should be around 0.7
    flagAnt[[ant for ant_index,ant in enumerate(useAntMap) if abs(Aeff[ant_index] - np.median(Aeff)) > 0.25]] *= 0.0    # flag antennas with > 25% departure 
flagBL = flagAnt[ANT0[0:blNum]]* flagAnt[ANT1[0:blNum]]
useAntMap = np.where(flagAnt > 0.0)[0].tolist()
useBlMap = np.where(flagBL > 0.0)[0].tolist()
useAntNum, useBlNum = len(useAntMap), len(useBlMap)
text_sd = '  Usable antennas (%d) : ' % (len(useAntMap))
for ants in antList[useAntMap].tolist(): text_sd = text_sd + ants + ' '
print(text_sd); logfile.write('#' + text_sd + '\n')
if useAntNum < 4: exit()
#-------- Select reference antenna
print('-----Select reference antenna')
timeStamp, UVW = GetUVW(msfile, BandbpSPW['spw'][0], polCalScan)
uvw = np.mean(UVW, axis=2); uvDist = np.sqrt(uvw[0]**2 + uvw[1]**2)
refantID = bestRefant(uvDist, [ant for ant in useAntMap if antDia[ant] >= np.percentile(antDia, 80)]) if refant == '' or refant not in antList[useAntMap] else np.where(antList == refant)[0][0]
text_sd = '  Use %s as refant' % (antList[refantID])
print(text_sd); logfile.write('#' + text_sd + '\n')
#-------- Remap baselines in canonical ordering
antMap = [refantID] + [ant for ant in useAntMap if ant != refantID]
blMap, blInv = map(list, zip(*[Ant2BlD(antMap[ANT0[bl_index]], antMap[ANT1[bl_index]]) for bl_index in range(useBlNum)]))
ant0, ant1 = ANT0[0:useBlNum], ANT1[0:useBlNum]
#-------- Load all-scan visibilities
for scan_index, scan in enumerate(BandScanList):
    timeStamp, UVW = GetUVW(msfile, BandbpSPW['spw'][0], scan)
    scanDic[scan]['UVW'] = UVW[:,blMap]
    for spw_index, spw in enumerate(BandbpSPW['spw']):
        Xspec = CrossPolBL(XspecList[spw_index][scan_index][:,:,blMap], blInv)  # Phase inversion for inverted baselines
        XspecList[spw_index][scan_index] = Xspec
#-------- Bandpass table
print('-----Bandpass to align SPWs and polarization')
#-------- Bandpass using polCalScan
FreqList, BPList, spwGainList = [], [], []
for spw_index, spw in enumerate(BandbpSPW['spw']):
    Xspec = XspecList[spw_index][BandScanList.index(polCalScan)]
    BP_ant, BPCaledXYSpec, XYdelay, Gain, XYsnr = CrossPolBP(Xspec)
    BPList = BPList + [BP_ant]
    spwGainList = spwGainList + [Gain]
#-------- SPW phase offsets
spwTwiddle = SPWalign(np.array(spwGainList))    # spwTwiddle[ant, pol, spw]
#-------- Phase-aligned bandpass table
print('-----SPW-aligned bandpass in scan %d : %s' % (polCalScan, scanDic[polCalScan]['source']))
for spw_index, spw in enumerate(BandbpSPW['spw']): BPList[spw_index] = (BPList[spw_index].transpose(2,0,1) * spwTwiddle[:,:,spw_index]).transpose(1,2,0)
pp = PdfPages('BP-%s-%s.pdf' % (prefix,BandName))
plotSP(pp, prefix, antList[antMap], BandbpSPW['spw'], BandbpSPW['freq'], BPList)
os.system('rm -rf B0')
bandpass(msfile, caltable='B0', spw=','.join(map(str, BandbpSPW['spw'])), scan=str(polCalScan), refant=antList[refantID], solnorm=True, minblperant=3)
#-------- SPW-combined phase calibration and check coherence for each scan
print('-----Antenna-based gain correction')
text_sd = '        coherence loss % :'
for antName in antList[antMap]: text_sd = text_sd + ' ' +  antName
print(text_sd)
for scan_index, scan in enumerate(BandScanList):
    text_sd = '----- Scan%3d %10s :' % (scan, scanDic[scan]['source'])
    chAvgList = []
    for spw_index, spw in enumerate(BandbpSPW['spw']):
        chRange = BandbpSPW['chRange'][spw_index]
        BP_ant = BPList[spw_index]
        BPcaledSpec = XspecList[spw_index][scan_index][[0,3]].transpose(3,2,0,1) / (BP_ant[ant0]* BP_ant[ant1].conjugate())
        chAvgVis    = np.mean(BPcaledSpec[:,:,:,chRange], axis=(2,3))   # chAvgVis[time, bl]
        chAvgList = chAvgList + [chAvgVis]
    #-------- Filter outliners out
    scanGain = np.zeros([BP_ant.shape[0], chAvgVis.shape[0]], dtype=complex)
    bl_vis = np.mean(np.array(chAvgList), axis=0).T
    #outLierVis = np.where(abs(bl_vis) > 10.0* np.median(abs(bl_vis)))
    outLierVis = np.where(abs(bl_vis) > 10.0* np.percentile(abs(np.mean(bl_vis, axis=1)), 80))
    scanFlag = list(set(range(bl_vis.shape[1])) - set(np.unique(outLierVis[1]))); scanFlag.sort()
    if len(scanFlag) > 1:
        scanGain[:,scanFlag] = np.apply_along_axis(gainComplex, 0, bl_vis[:,scanFlag])
        scanAnt = np.where(abs( np.log( np.mean(abs(scanGain[:,scanFlag]), axis=1) / np.median(abs(scanGain[:,scanFlag])))) < np.log(10))[0].tolist()
    #-------- Store to scanDic
    scanDic[scan]['scanFlag'] = scanFlag
    scanDic[scan]['scanAnt']  = scanAnt
    scanDic[scan]['Gain']   = scanGain
    coh = abs(np.mean(scanGain[:,scanFlag], axis=1)) / np.mean(abs(scanGain[:,scanFlag]), axis=1)
    for ant_index, ant in enumerate(antList[antMap]): text_sd = text_sd + ' %.2f' % (100.0* (1.0 - coh[ant_index]))
    print(text_sd)
#-------- Scan-by-scan bandpass and XY delay with SNR
BPavgScanList = []  # scans used in bandpass averaging
text_sd = '-----Scan-by-scan BP     :'
for spw_index, spw in enumerate(BandbpSPW['spw']): text_sd = text_sd + '     SPW%2d     ' % (spw)
print(text_sd)
text_sd = '----- XY delay           :'
for spw_index, spw in enumerate(BandbpSPW['spw']): text_sd = text_sd + '  [ns]  ( SNR )'
print(text_sd)
for scan in QSOscanList:
    scan_index = BandScanList.index(scan)
    text_sd = '----- Scan%3d %10s :' % (scan, scanDic[scan]['source'])
    scanFlag  = scanDic[scan]['scanFlag']
    if len(scanFlag) == 0: continue
    scanPhase = scanDic[scan]['Gain'][:,scanFlag] / abs(scanDic[scan]['Gain'][:,scanFlag])
    BPSPWList, XYSPWList, XYsnrList = [], [], []
    for spw_index, spw in enumerate(BandbpSPW['spw']):
        chNum   = BandbpSPW['chNum'][spw_index]
        Xspec = XspecList[spw_index][scan_index][:,:,:,scanFlag]         # Xspec[pol, ch, bl, time]
        XPspec = np.mean(Xspec/ (scanPhase[ant0]* scanPhase[ant1].conjugate()), axis=3)   # antenna-based phase correction
        BP_ant = np.array([gainComplexVec(XPspec[0].T), gainComplexVec(XPspec[-1].T)])
        #---- Amplitude normalization
        for pol_index in [0,1]:
            BP_eq_gain = np.mean(abs(BP_ant[pol_index][:,BandbpSPW['chRange'][spw_index]]), axis=1)
            BP_ant[pol_index] = (BP_ant[pol_index].T / BP_eq_gain).T
        BPCaledXspec = XPspec.transpose(0, 2, 1)/(BP_ant[polYindex][:,ant0]* BP_ant[polXindex][:,ant1].conjugate())
        BPCaledXYSpec = np.mean(BPCaledXspec[1], axis=0) +  np.mean(BPCaledXspec[2], axis=0).conjugate()
        XYdelay, XYsnr = delay_search( BPCaledXYSpec[BandbpSPW['chRange'][spw_index]] )
        XYdelay = (float(chNum) / float(len(BandbpSPW['chRange'][spw_index])))* XYdelay
        text_sd = text_sd + ' %6.3f (%5.1f)' % (0.5* XYdelay/(BandbpSPW['BW'][spw_index]*1.0e-9), XYsnr)
        if np.median(scanDic[scan]['EL']) < ELshadow: XYsnr *= (np.median(scanDic[scan]['EL']) / ELshadow / 2)**2
        XYsnrList = XYsnrList + [XYsnr]
        BPSPWList = BPSPWList + [BP_ant.transpose(1,0,2)]
        XYSPWList = XYSPWList + [BPCaledXYSpec]
    print(text_sd)
    scanDic[scan]['BP'] = BPSPWList
    scanDic[scan]['XY'] = XYSPWList
    scanDic[scan]['XYsnr'] = XYsnrList
    BPavgScanList = BPavgScanList + [scan]
#-------- Average bandpass
for spw_index, spw in enumerate(BandbpSPW['spw']):
    #print('XY List : %s spw=%d, spw_index=%d' % (BandName, spw, spw_index))
    chRange = BandbpSPW['chRange'][spw_index]
    XYsnrList = [scanDic[scan]['XYsnr'][spw_index] for scan in BPavgScanList]
    XYrefscan = BPavgScanList[XYsnrList.index(max(XYsnrList))]
    refXYspec = scanDic[XYrefscan]['XY'][spw_index]
    XY = 0.0* refXYspec
    BP = 0.0* scanDic[XYrefscan]['BP'][spw_index]
    for scan_index, scan in enumerate(BPavgScanList):
        if scanDic[scan]['XYsnr'][spw_index] < 1.0: continue
        XYspec = scanDic[scan]['XY'][spw_index]
        XYcorr = XYspec[chRange].dot(refXYspec[chRange].conjugate())
        XY = XY + abs(XYcorr)* np.sign(XYcorr.real)* XYspec
    BPweight = [abs(np.mean(scanDic[scan]['Gain'])) for scan in BPavgScanList]
    for scan_index, scan in enumerate(BPavgScanList):
        if len(scanDic[scan]['scanFlag']) < 3: continue
        BP = BP + BPweight[scan_index]* scanDic[XYrefscan]['BP'][spw_index]
    BP = BP / np.sum(BPweight)
    BP[:,1] *= (XY / abs(XY))
    BPSPWList[spw_index] = BP
    #---- Save into CASA caltable
    if os.path.isfile('B0'):
        tb.open('B0', nomodify=False)
        SPWQ = tb.query('SPECTRAL_WINDOW_ID == %d'%(spw))
        BPtable = SPWQ.getcol('CPARAM')
        for ant_index, ant in enumerate(antMap): BPtable[:,:,ant] = BPSPWList[spw_index][ant_index]
        SPWQ.putcol('CPARAM', BPtable)
        SPWQ.close()
        tb.close()
pp = PdfPages('BP-%s-%s-%d.pdf' % (prefix, BandName, 0))
plotSP(pp, prefix, antList[antMap], BandbpSPW['spw'], BandbpSPW['freq'], BPSPWList, 0.0, 1.2, True)
#-------- XY sign in polCalScan
for spw_index, spw in enumerate(BandbpSPW['spw']):
    BP_ant = BPSPWList[spw_index].transpose(1,2,0)
    BPCaledXY = 0.0 + 0.0j
    for scan_index, scan in enumerate(BPavgScanList):
        if scanDic[scan]['XYsnr'][spw_index] < 1.0: continue
        scanFlag  = scanDic[scan]['scanFlag']
        scanPhase = scanDic[scan]['Gain'][:,scanFlag] / abs(scanDic[scan]['Gain'][:,scanFlag])
        Xspec = np.mean(XspecList[spw_index][list(scanDic.keys()).index(scan)][:,:,:,scanFlag]*(scanPhase[ant1]* scanPhase[ant0].conjugate()), axis=3)
        BPCaledXspec = Xspec / (BP_ant[polYindex][:,:,ant0]* BP_ant[polXindex][:,:,ant1].conjugate())
        BPCaledXY   += scanDic[scan]['UCmQS'][spw_index]* (np.mean(BPCaledXspec[1][chRange]) +  np.mean(BPCaledXspec[2][chRange].conjugate()))
    XYphase = np.angle(BPCaledXY)
    XYsign = np.sign(BPCaledXY.real) if 'givenXYsign' not in locals() else np.sign(givenXYsign)
    text_sd = 'SPW[%d] : XY phase = %6.1f [deg] sign = %3.0f' % (spw, 180.0*XYphase/np.pi, XYsign)
    logfile.write('# ' + text_sd +'\n'); print(text_sd)
    BPSPWList[spw_index][:,1] *= XYsign
#-------- Apply Bandpass and Phase Correction
for scan_index, scan in enumerate(BandScanList):
    scanFlag  = scanDic[scan]['scanFlag']
    if len(scanFlag) == 0: continue
    scanPhase = scanDic[scan]['Gain'][:,scanFlag] / abs(scanDic[scan]['Gain'][:,scanFlag])
    for spw_index, spw in enumerate(BandbpSPW['spw']):
        # print('---Applying Gain and Bandpass Correction for scan %d, spw %d' % (scan, spw))
        BP_ant = BPSPWList[spw_index].transpose(1,2,0)
        Xspec = XspecList[spw_index][scan_index][:,:,:,scanFlag]* (scanPhase[ant1]* scanPhase[ant0].conjugate())
        XspecList[spw_index][scan_index] = (Xspec.transpose(3, 0, 1, 2) / (BP_ant[polYindex][:,:,ant0]* BP_ant[polXindex][:,:,ant1].conjugate())).transpose(1,2,3,0)
#-------- Aperture Efficiencies Determination using Solar System Objects
if 'Apriori' not in locals(): Apriori = False
if not Apriori:
    for scan_index, scan in enumerate(BandScanList):
        if len(scanDic[scan]['scanFlag']) == 0: continue
        if scan in QSOscanList : continue              # filter QSO out
        if scanDic[scan]['source'] not in SSOCatalog : continue    # not a usable SSO
        uvw = np.mean(scanDic[scan]['UVW'], axis=2) #; uvDist = np.sqrt(uvw[0]**2 + uvw[1]**2)
        FscaleDic[scanDic[scan]['source']] = SSOAe(antList[antMap], BandbpSPW, uvw, scanDic[scan], SSODic, [XspecList[spw_index][scan_index][0::3] for spw_index,spw in enumerate(BandbpSPW['spw'])])
SSOList = list(FscaleDic.keys())
for SSO in SSOList:
    if (FscaleDic[SSO] == []) or (FscaleDic[SSO] == None): del FscaleDic[SSO]
#-------- WgSum : to judge SSO flux calibration or a priori (SEFD)
WgSum = np.sum( [np.median(np.array(FscaleDic[SSO]['Wg'])) for SSO in FscaleDic.keys()] )
Aeff =  averageAe(FscaleDic, BandbpSPW['spw']) if WgSum > 0.01 else np.array([etaA[:,antMap] for spw in BandbpSPW['spw']]).transpose(2,1,0)
text_sd = ' Aeff: '
for spw_index, spw in enumerate(BandbpSPW['spw']): text_sd = text_sd + 'SPW%02d-X SPW%02d-Y ' % (spw, spw)
fluxCalText = ''
for source in FscaleDic.keys() : fluxCalText = fluxCalText + ' %s,' % (source)
text_sd = text_sd + fluxCalText
logfile.write(text_sd +'\n'); print(text_sd)
logjy = open(prefix + '-' + BandName + '-JyK.log', 'w')
timeLabel = qa.time('%fs' % np.median(scanDic[polCalScan]['mjdSec']), form='ymd')[0]
for ant_index, ant in enumerate(antList[antMap]):
    if np.min(Aeff[ant_index]) < 1.0e-6: continue
    text_sd = '%s : ' % (ant)
    for spw_index, spw in enumerate(BandbpSPW['spw']):
        for pol_index, polName in enumerate(['X','Y']):
            text_sd = text_sd + '  ----- ' if Aeff[ant_index][pol_index][spw_index] < 0.1 else text_sd + '  %4.1f%% ' % (100.0* Aeff[ant_index][pol_index][spw_index])
            text_jy = '%s %s %d %s %f %s %s %e %e %5.1f' % (prefix, ant, spw, polName, 2.0* kb / (0.25* Aeff[ant_index][pol_index][spw_index]* np.pi*antDia[ant_index]**2), timeLabel, BandName, np.median(BandbpSPW['freq'][spw_index]), abs(BandbpSPW['BW'][spw_index]), tempAtm); logjy.write(text_jy + '\n')
    logfile.write(text_sd +'\n'); print(text_sd)
logjy.write('\n'); logjy.close()
#-------- Gain transfer and equalization using check scan
QSONonShadowScanList = [scan for scan in QSOscanList if np.median(scanDic[scan]['EL']) > ELshadow]
if (len(QSONonShadowScanList) > 0) and (checkScan not in QSONonShadowScanList):
    checkScan = QSONonShadowScanList[np.argmax([scanDic[scan]['I'] for scan in QSONonShadowScanList])]
scan_index = list(scanDic.keys()).index(checkScan)
Aeff = np.array([AeTransfer(np.mean(XspecList[spw_index][scan_index][0::3][:,chRange], axis=(1,3)), Aeff[:,:,spw_index], antDia[antMap]) for spw_index, spw in enumerate(BandbpSPW['spw'])]).transpose(1,2,0)
#-------- Stokes visibilities for each scan
text_Stokes = np.repeat('',len(BandbpSPW['spw'])).tolist()
#-------- Store Stokes parameters into scanDic
for scan_index, scan in enumerate(scanDic.keys()):
    scanFlag  = scanDic[scan]['scanFlag']
    flagAnt = np.ones(useAntNum)
    shadowAnts = indexList(scanDic[scan]['Shadow'], antList[antMap])
    flagAnt[shadowAnts] *= 0.1
    flagBL = flagAnt[ANT0[0:useBlNum]]* flagAnt[ANT1[0:useBlNum]]
    unShadowBL = [bl for bl in list(range(len(flagBL))) if flagBL[bl] > 0.5]
    uvw = np.mean(scanDic[scan]['UVW'], axis=2); uvDist = np.sqrt(uvw[0]**2 + uvw[1]**2)
    if len(scanFlag) == 0: continue
    sourceName = scanDic[scan]['source']
    text_src  = ' %02d %010s EL=%4.1f deg' % (scan, sourceName, 180.0* np.median(scanDic[scan]['EL'])/np.pi)
    timeLabel = qa.time('%fs' % np.median(scanDic[scan]['mjdSec']), form='ymd')[0] + ' SA=%.1f' % (scanDic[scan]['SA']) + ' deg.'
    StokesVisList, visChavList = [], []
    ScanFlux, ErrFlux, ScanSlope = np.zeros([len(BandbpSPW['spw']), 4]), np.zeros([len(BandbpSPW['spw']), 4]), np.zeros(len(BandbpSPW['spw']))
    for spw_index, spw in enumerate(BandbpSPW['spw']):
        text_Stokes[spw_index] = ' SPW%02d %5.1f GHz ' % (spw, 1.0e-9* np.median(BandbpSPW['freq'][spw_index]))
        visChav = GainScale(Aeff[:,:,spw_index], antDia[antMap], np.mean(XspecList[spw_index][scan_index][:,chRange], axis=1))
        visChavList = visChavList + [visChav]
        StokesVis = Vis2Stokes(visChav, Dcat[antMap][:,:,spw_index], scanDic[scan]['PA'][scanFlag])
        StokesVisList = StokesVisList + [StokesVis]
        #-------- SSO visibility to correct by model
        if sourceName in FscaleDic.keys(): StokesVis *= (SSODic[sourceName][1][spw_index] / FscaleDic[sourceName]['model'][spw_index])
        #-------- Linear regression to determine zero-spacing visibilities
        ScanFlux[spw_index], ScanSlope[spw_index], ErrFlux[spw_index] = lmStokes(StokesVis[:,unShadowBL], uvDist[unShadowBL])
        for pol_index in list(range(4)):
            text_Stokes[spw_index] = text_Stokes[spw_index] + ' %7.4f (%.4f) ' % (ScanFlux[spw_index, pol_index], ErrFlux[spw_index, pol_index])
        text_Stokes[spw_index] = text_Stokes[spw_index] + '%6.3f   %6.1f ' % (100.0* np.sqrt(ScanFlux[spw_index, 1]**2 + ScanFlux[spw_index, 2]**2)/ScanFlux[spw_index, 0], np.arctan2(ScanFlux[spw_index, 2],ScanFlux[spw_index, 1])*90.0/np.pi)
    #---- Update scanDic record entry
    CS, SN = np.cos(2.0* scanDic[scan]['PA']), np.sin(2.0* scanDic[scan]['PA'])
    scanDic[scan]['blFlag'] = unShadowBL
    scanDic[scan]['ScanFlux'] = ScanFlux        # ScanFlux[spw,pol]
    scanDic[scan]['QCpUS'] = [ScanFlux[spw_index, 1]* CS + ScanFlux[spw_index, 2]* SN for spw_index, spw in enumerate(BandbpSPW['spw'])]
    scanDic[scan]['UCmQS'] = [ScanFlux[spw_index, 2]* CS - ScanFlux[spw_index, 1]* SN for spw_index, spw in enumerate(BandbpSPW['spw'])]
    scanDic[scan]['visChav'] = visChavList
    scanDic[scan]['StokesVis'] = StokesVisList
    scanDic[scan]['uvSlope'] = ScanSlope
    uvMin, uvMax, IMax = min(uvDist), max(uvDist), max(ScanFlux[:,0])
    refFreq = 1.0e-9* np.mean(BandbpSPW['freq'])
    relFreq = 1.0e-9* np.array([np.median( BandbpSPW['freq'][spw_index]) for spw_index, spw in enumerate(BandbpSPW['spw'])]) - refFreq
    text_mean = ' mean  %5.1f GHz ' % (refFreq)
    pflux, pfluxerr = np.zeros(4), np.zeros(4)
    for pol_index in list(range(4)):
        sol, solerr = linearRegression(relFreq, ScanFlux[:, pol_index], ErrFlux[:, pol_index] )
        pflux[pol_index], pfluxerr[pol_index] = sol[0], solerr[0]
        text_mean = text_mean + ' %7.4f (%.4f) ' % (pflux[pol_index], pfluxerr[pol_index])
    text_mean = text_mean + '%6.3f   %6.1f' % (100.0* np.sqrt(pflux[1]**2 + pflux[2]**2)/pflux[0], np.arctan2(pflux[2],pflux[1])*90.0/np.pi) 
    text_sd = text_src + ' ' + timeLabel
    logfile.write(text_sd +'\n'); print(text_sd)
    text_sd = ' SPW  Frequency     I                 Q                 U                 V             |  %Pol     EVPA '; logfile.write(text_sd + '\n'); print(text_sd)
    text_sd = ' --------------------------------------------------------------------------------------------------------'; logfile.write(text_sd + '\n'); print(text_sd)
    for spw_index,spw in enumerate(BandbpSPW['spw']): logfile.write(text_Stokes[spw_index] + '\n'); print(text_Stokes[spw_index])
    text_sd = ' --------------------------------------------------------------------------------------------------------'; logfile.write(text_sd + '\n'); print(text_sd)
    if len(BandbpSPW['spw']) > 1: logfile.write(text_mean + '\n'); print(text_mean)
    text_sd = ' UV_min_max  %6.1f  %6.1f ' % (uvMin, uvMax); logfile.write(text_sd + '\n'); print(text_sd)
    text_ingest = '%10s, NE, NE, NE, NE, %e, %6.3f, %6.4f, %6.3f, %6.4f, %5.1f, %5.1f, NE, NE, %s, %s, %s\n' % (scanDic[scan]['source'], np.median(BandbpSPW['freq'][spw_index]), pflux[0], np.sqrt(0.0004*pflux[0]**2 + pfluxerr[0]**2), np.sqrt(pflux[1]**2 + pflux[2]**2)/pflux[0], np.sqrt(pfluxerr[1]**2 + pfluxerr[2]**2)/pflux[0], np.arctan2(pflux[2],pflux[1])*90.0/np.pi, np.sqrt(pfluxerr[1]**2 + pfluxerr[2]**2)/np.sqrt(pflux[1]**2 + pflux[2]**2)*90.0/np.pi, timeLabel.replace('/','-'), fluxCalText, prefix)#; ingestFile.write(text_ingest)
#-------- Plot Stokes parameters vs uv distance
if PLOTFL: pp = PdfPages('FL_' + prefix + '_' + BandName + '.pdf'); plotFL(pp, scanDic, BandbpSPW)
#-------- Review D-term
Dterm = np.zeros([useAntNum, len(BandbpSPW['spw']), 2], dtype=complex)
DtermDic = {'mjdSec': np.median(timeStamp)}
DtermDic['Freq'] = [np.median(BandbpSPW['freq'][spw_index]) for spw_index, spw in enumerate(BandbpSPW['spw'])]
text_sd = 'D-term        '; logfile.write(text_sd); print(text_sd, end='')
for spw_index, spw in enumerate(BandbpSPW['spw']):
    IList, QCpUSList, UCmQSList, visChavList = [], [], [], []
    for scan_index, scan in enumerate(QSOscanList):
        scanFlag  = scanDic[scan]['scanFlag']
        if len(scanFlag) == 0: continue
        if min(scanDic[scan]['EL']) < ELshadow : continue
        IList       = IList + [scanDic[scan]['ScanFlux'][spw_index,0] for time_index in scanFlag]
        QCpUSList   = QCpUSList + scanDic[scan]['QCpUS'][spw_index][scanFlag].tolist()
        UCmQSList   = UCmQSList + scanDic[scan]['UCmQS'][spw_index][scanFlag].tolist()
        visChavList = visChavList + scanDic[scan]['visChav'][spw_index].transpose(2, 0, 1).tolist()
    Dterm[:,spw_index, 0], Dterm[:,spw_index,1] = VisMuiti_solveD(np.array(visChavList).transpose(1, 2, 0), np.array(QCpUSList), np.array(UCmQSList), Dcat[antMap,0,spw_index], Dcat[antMap,1,spw_index], np.array(IList))
#---- Display D-terms
for ant_index, ant in enumerate(antList[antMap]): DtermDic[ant] = Dterm[ant_index]
for spw_index, spw in enumerate(BandbpSPW['spw']):
    text_sd = '    SPW%02d                    ' % (spw);  logfile.write(text_sd); print(text_sd, end=' ')
logfile.write('\n'); print('')
text_sd = 'Ant  '; logfile.write(text_sd); print(text_sd, end='')
for spw_index, spw in enumerate(BandbpSPW['spw']):
    text_sd = '       Dx             Dy     ';  logfile.write(text_sd); print(text_sd, end='')
logfile.write('\n'); print('')
for ant_index, ant in enumerate(antList[antMap]):
    text_sd = '%s  ' % (ant)
    text_fd = text_sd
    for spw_index, spw in enumerate(BandbpSPW['spw']):
        for pol_index in [0,1]:
            if abs(Dterm[ant_index,spw_index,pol_index]) > 0.1:
                text_sd = text_sd + '\033[91m %+.3f%+.3fi \033[0m' % (Dterm[ant_index,spw_index,pol_index].real, Dterm[ant_index,spw_index,pol_index].imag)
            else:
                text_sd = text_sd + ' %+.3f%+.3fi ' % (Dterm[ant_index,spw_index,pol_index].real, Dterm[ant_index,spw_index,pol_index].imag)
            #
            text_fd = text_fd + ' %+.3f%+.3fi ' % (Dterm[ant_index,spw_index,pol_index].real, Dterm[ant_index,spw_index,pol_index].imag)
        #
    #
    logfile.write(text_fd + '\n'); print(text_sd)
#----
logfile.close()
#-------- Plot QUXY
for scan_index, scan in enumerate(scanDic.keys()):
    if 'StokesVis' in scanDic[scan].keys():
        scanDic[scan]['scanVis'] = np.mean(np.array(scanDic[scan]['StokesVis']), axis=2).real.T
        scanDic[scan]['visChav'] = np.mean(np.array(scanDic[scan]['visChav']), axis=(0, 2))
if PLOTQU: pp = PdfPages('QU-%s-%s.pdf' % (prefix,BandName)); plotQUXY(pp, scanDic)
if PLOTMAP: pp = PdfPages('MP_' + prefix + '_' + BandName + '.pdf'); plotResidualMap(pp, scanDic, BandbpSPW)
msmd.close()
