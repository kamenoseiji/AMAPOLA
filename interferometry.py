import os
import re
import numpy as np
import scipy
from scipy.interpolate import UnivariateSpline
import datetime
import urllib.request, urllib.error
import ssl
from casatools import table as tbtool
from casatools import msmetadata as msmdtool
from casatools import quanta as qatool
tb = tbtool()
msmd = msmdtool()
qa = qatool()
#-------- Constants
Tcmb = 2.725        # CMB temperature
kb   = 1.38064852e3 # Boltzman constant (* 1e26 for Jy scaling)
RADDEG = 180.0 / np.pi        # 57 deg per rad
speed_of_light = 299792458  # [m/s]
MJD_EPOCH = datetime.datetime(1858, 11, 17, 0, 0, 0, tzinfo=datetime.timezone.utc)
ALMA_long= -67.755/180.0* np.pi     # ALMA AOS Longitude
ALMA_lat = -23.029/180.0* np.pi     # ALMA AOS Latitude
BANDPA = [0.0, 45.0, -45.0, 80.0, -80.0, 45.0, -45.0, 36.45, 90.0, -90.0, 0.0]   # X-pol orientation for Band-1, 2, 3, 4, 5, 6, 7, 8, 9, and 10
BANDFQ = [0.0, 40.0, 75.0, 97.5, 154.9, 183.0, 233.0, 343.5, 460.0, 650.0, 870.0]   # Standard frequency [GHz]
BANDFQL= [0.0, 35.0, 67.0, 84.0, 125.0, 158.0, 211.0, 275.0, 385.0, 602.0, 787.0]   # Band lower limit [GHz]
BANDFQU= [35.0,50.0, 84.0,116.0, 158.0, 211.0, 275.0, 373.0, 500.0, 720.0, 950.0]   # Band upper limit [GHz] but Band-2 and 4 were modified to avoid overlap
#-------- Indexing
KERNEL_BL = (np.arange(64)*np.arange(1,65)/2).astype(int)
def Bl2Ant(bl_index):     # Baseline -> antenna indexing (canonical ordering)
    ant1 = max(np.where(KERNEL_BL<= bl_index)[0]) + 1
    return ant1, int(bl_index - KERNEL_BL[ant1 - 1])
#
ANT0 = [Bl2Ant(bl_index)[0] for bl_index in range(2016)]
ANT1 = [Bl2Ant(bl_index)[1] for bl_index in range(2016)]
def Ant2Bl(station1, station2):     # Antenna -> baseline index (without autocorr)
    antenna1, antenna2 = (station1, station2) if station1 > station2 else (station2, station1)
    return int(antenna1* (antenna1 - 1)/2 + antenna2)
#   
def Ant2BlD(station1, station2):    # Antenna -> baseline index and direction (True if inverted)
    antenna1, antenna2 = (station1, station2) if station1 > station2 else (station2, station1)
    return int(antenna1* (antenna1 - 1)/2 + antenna2), (station1 < station2) 
#   
def indexList(refArray, motherArray):     # Compare two arrays and return matched index
    return [np.where(np.array(motherArray) == currentItem)[0][0] for currentItem in refArray if currentItem in motherArray]
# 
def subArrayIndex(Flag, refant):          #-------- SubArray Indexing
    blNum = len(Flag); antNum = Bl2Ant(blNum)[0]
    ant0, ant1 = np.array(ANT0[0:blNum]), np.array(ANT1[0:blNum])
    kernelBL = np.where(ant0 == refant)[0].tolist() + np.where(ant1 == refant)[0].tolist()
    flagIndex = np.where(Flag[kernelBL] > 0.5)[0].tolist()
    useKernelBL = np.array(kernelBL)[flagIndex].tolist()
    SAantennas = list(set(np.append(ant0[useKernelBL], ant1[useKernelBL])))
    SAantMap = [refant] + sort(np.array(list(set(SAantennas) - set([refant])))).tolist()
    SAantNum = len(SAantennas); SAblNum = int(SAantNum* (SAantNum - 1)/2)
    SAblMap, SAblInv = Ant2BlD(SAantMap[np.array(ant0)[0:SAblNum]], SAantMap[np.array(ant1)[0:SAblNum]])
    return SAantMap, SAblMap, SAblInv
#
def bestRefant(uvDist, useantList=[]):      # Select the best reference antenna
    blNum = len(uvDist)
    antNum = Bl2Ant(blNum)[0]
    if useantList == []: useantList = list(range(antNum))
    distOrder = np.argsort(uvDist)
    blCounter = np.zeros(antNum) 
    for blID in distOrder:
        ant1, ant0 = Bl2Ant(blID)
        if ant1 in useantList: blCounter[ant1] += 1
        if ant0 in useantList: blCounter[ant0] += 1
        if np.max(blCounter) > 1: break
    return np.argmax(blCounter)
#
def GetBaselineIndex(msfile, spwID, scanID):
    data_desc_id = SPW2DATA_DESC_ID(msfile, spwID)
    Out='DATA_DESC_ID == %d && SCAN_NUMBER == %d' % (data_desc_id, scanID)
    tb.open(msfile)
    antXantYspw = tb.query(Out)
    timeXY = antXantYspw.getcol('TIME')
    time0_index = np.where(timeXY == timeXY[0])[0].tolist()
    Antenna1, Antenna2 = antXantYspw.getcol('ANTENNA1')[time0_index], antXantYspw.getcol('ANTENNA2')[time0_index]
    tb.close()
    return Antenna1, Antenna2
#
def SPW2DATA_DESC_ID(msfile, spwID):
    if os.path.isdir( msfile + '/' + 'DATA_DESCRIPTION' ):
        tb.open(msfile + '/' + 'DATA_DESCRIPTION')
        data_desc_id, spw_index = -1,-1
        while( spw_index !=  spwID ):
            data_desc_id += 1
            spw_index = tb.getcell('SPECTRAL_WINDOW_ID', data_desc_id)
        #
        tb.close()
        return data_desc_id
    else:
        return spwID
#
def CrossCorrAntList(Antenna1, Antenna2):
    blList = np.where(Antenna1 != Antenna2)[0].tolist()
    return np.unique(Antenna1[blList]).tolist() + [np.unique(Antenna2[blList]).tolist()[-1]]
#
def corrAddress(Antenna1, Antenna2):
    UseAntList = CrossCorrAntList(Antenna1, Antenna2)
    antNum = len(UseAntList)
    blNum  = int(antNum* (antNum - 1)/2)
    xcorr_index = list(range(blNum))
    for bl_index in list(range(blNum)):
        ant1, ant0 = Bl2Ant(bl_index)
        xcorr_index[bl_index] = list(set(np.where(Antenna2 == UseAntList[ant1])[0]) & set(np.where(Antenna1 == UseAntList[ant0])[0]))[0]
    return UseAntList, xcorr_index
#
#-------- Access to MS
def GetTemp(msfile):                        # Air temperature
    Out = msfile + '/WEATHER'
    tb.open(Out)
    temp = tb.getcol('TEMPERATURE')
    tb.close()
    if len(temp) == 0:
        return GetLoadTemp(msfile, 0,0)[0]
    else:
        return np.median(temp)
#  
def GetLoadTemp(msfile, AntID, spw):        # ACD load temperature
    Out = msfile + '/' + 'CALDEVICE'
    tb.open(Out)
    Condition = 'ANTENNA_ID == %d && SPECTRAL_WINDOW_ID == %d' % (AntID, spw)
    temp = tb.query(Condition).getcol('TEMPERATURE_LOAD')
    tb.close()
    return np.median(temp, axis=1) if np.ndim(temp) == 2 else temp
#
def GetAntName(msfile):                     # Antenna name list
    tb.open(msfile+'/'+'ANTENNA')
    namelist = tb.getcol("NAME")
    tb.close() 
    return namelist
#
def GetAntPos(msfile):                      # Antenna position
    tb.open(msfile+'/'+'ANTENNA')
    antPos = tb.getcol("POSITION")
    tb.close()
    return antPos
#
def GetAntD(antList):                       # Antenna diameter list
    antD = 12.0* np.ones(len(antList))
    for ant_index, antName in enumerate(antList):
        if antName.find('C') > -1: antD[ant_index] = 7.0
    return antD
# 
def GetAzEl(msfile):                        # Antenna Az, El
    Out = msfile + '/' + 'POINTING'
    tb.open(Out)
    Direction=tb.getcol("DIRECTION")
    Time     = tb.getcol("TIME")
    AntID    = tb.getcol("ANTENNA_ID")
    tb.close()
    return Time, AntID, Direction[0,0], Direction[1,0]
# 
def GetBandNames(msfile, atmSPWs=[]):       # Receiver band name
    if len(atmSPWs) < 1: atmSPWs = GetAtmSPWs(msfile)
    msmd.open(msfile)
    atmspwNames = msmd.namesforspws(atmSPWs)
    atmBandNames = [re.findall(r'RB_..', spwName)[0] for spwName in atmspwNames if 'RB_' in spwName]
    if len(atmBandNames) < len(atmSPWs):                # spwName is blank in early observations, then guess band name by observing frequency
        tb.open(msfile + '/' + 'SPECTRAL_WINDOW')
        spwFreqList = [1.0e-9* np.median(tb.getcell("CHAN_FREQ", spwID)) for spwID in atmSPWs]
        tb.close()
        atmBandNames = ['RB_%02d' % [band_index for band_index,freq in enumerate(BANDFQL) if spwFreq > freq and spwFreq < BANDFQU[band_index]][0] for spwFreq in spwFreqList]
    msmd.close(); msmd.done()
    return atmBandNames 
#  
def GetBPcalSPWs(msfile):                   # SPWs used in bandpass scan
    msmd.open(msfile)
    bpSPWs  = msmd.spwsforintent("CALIBRATE_PHASE*").tolist(); bpSPWs.sort()
    if len(bpSPWs) == 0: bpSPWs  = msmd.spwsforintent("CALIBRATE_POLARIZATION*").tolist(); bpSPWs.sort()
    if len(bpSPWs) == 0: bpSPWs  = msmd.spwsforintent("CALIBRATE_FLUX*").tolist(); bpSPWs.sort()
    if len(bpSPWs) == 0: bpSPWs  = msmd.spwsforintent("CALIBRATE_BANDPASS*").tolist(); bpSPWs.sort()
    if len(bpSPWs) == 0: bpSPWs  = msmd.spwsforintent("CALIBRATE_DELAY*").tolist(); bpSPWs.sort()
    BPspwList = []
    for spw in bpSPWs:
        chNum, chWid, freq = GetChNum(msfile, spw)
        if chNum > 7:   BPspwList = BPspwList + [spw]   # Filter out WVR and CHAVG spectral windows     
    msmd.close()  
    return BPspwList
#
def GetAtmSPWs(msfile):                 # SPWs used in atmCal scan
    msmd.open(msfile)
    atmSPWs = list( (set(msmd.tdmspws()) | set(msmd.fdmspws())) & set(msmd.spwsforintent("CALIBRATE_ATMOSPHERE*")) ); atmSPWs.sort()
    if len(atmSPWs) == 0:
        atmSPWList = msmd.spwsforintent("CALIBRATE_ATMOSPHERE*").tolist()
        tb.open(msfile + '/' + 'SPECTRAL_WINDOW')
        for spwID in atmSPWList:
            if tb.getcell("NUM_CHAN", spwID) > 60: atmSPWs = atmSPWs + [spwID]
        tb.close()
    msmd.close()
    return atmSPWs
#
def GetPHSPWs(msfile):                  # SPWs used in phase calibration scans
    msmd.open(msfile)
    phSPWs  = msmd.spwsforintent("CALIBRATE_PHASE*").tolist(); phSPWs.sort()
    SPWnames= msmd.namesforspws(phSPWs)
    msmd.close()
    return [spw for spw_index, spw in enumerate(phSPWs) if 'FULL_RES' in SPWnames[spw_index] ]
#
def GetPHchavSPWs(msfile):
    msmd.open(msfile)
    phSPWs  = msmd.spwsforintent("CALIBRATE_PHASE*").tolist(); phSPWs.sort()
    SPWnames= msmd.namesforspws(phSPWs)
    msmd.close()
    return [spw for spw_index, spw in enumerate(phSPWs) if 'CH_AVG' in SPWnames[spw_index] ]
#
def GetSPWFreq(msfile, SPWdic):         # Frequency setups in SPWs
    msmd.open(msfile)
    TDMspwList = msmd.tdmspws().tolist()
    msmd.close()
    chNumList, chRangeList, BWList, FreqList = [], [], [], []
    for spw in SPWdic['spw']:
        chNum, chWid, freq = GetChNum(msfile, spw)
        chRange = list(range(int(0.1*chNum), int(0.9*chNum))) if spw in TDMspwList else list(range(chNum))
        chNumList = chNumList + [chNum]
        chRangeList = chRangeList + [chRange]
        BWList = BWList + [chNum* np.median(chWid)]
        FreqList = FreqList + [freq]
    SPWdic['freq']  = FreqList
    SPWdic['chNum'] = chNumList
    SPWdic['chRange'] = chRangeList
    SPWdic['BW']    = BWList
    return SPWdic
#
def GetChNum(msfile, spwID):            # SPW Channel information
    tb.open(msfile + '/' + 'SPECTRAL_WINDOW')
    chNum = tb.getcell("NUM_CHAN", spwID)
    chWid = tb.getcell("CHAN_WIDTH", spwID)
    freq  = tb.getcell("CHAN_FREQ", spwID)
    tb.close()
    return chNum, chWid, freq
#   
def GetOnSource(msfile):                # On-source scans
    msmd.open(msfile)
    OnScanList = np.sort(np.array(list(set(msmd.scansforintent("*CALIBRATE_POLARIZATION*")) | set(msmd.scansforintent("*CALIBRATE_AMPLI*")) | set(msmd.scansforintent("*CALIBRATE_BANDPASS*")) | set(msmd.scansforintent("*CALIBRATE_FLUX*")) | set(msmd.scansforintent("*OBSERVE_CHECK_SOURC*")) | set(msmd.scansforintent("*CALIBRATE_DELAY*")) | set(msmd.scansforintent("*CALIBRATE_PHASE*")) | set(msmd.scansforintent("*OBSERVE_TARGET*")))))
    msmd.close()
    return OnScanList
#
def GetTimerecord(msfile, ant1, ant2, spwID, PScan):    # Time stamps
    data_desc_id = SPW2DATA_DESC_ID(msfile, spwID)
    Out='ANTENNA1 == %d && ANTENNA2 == %d && DATA_DESC_ID == %d && SCAN_NUMBER == %d' % (ant1, ant2, data_desc_id, PScan)
    tb.open(msfile)
    antXantYspw = tb.query(Out)
    interval=antXantYspw.getcol('INTERVAL')
    timeXY = antXantYspw.getcol('TIME')
    tb.close()
    return interval, timeXY
#
def GetUVW(msfile, spwID, scanID):      # UVW coordinates
    Antenna1, Antenna2 = GetBaselineIndex(msfile, spwID, scanID)
    acorr_index, xcorr_index = corrAddress(Antenna1, Antenna2)
    data_desc_id = SPW2DATA_DESC_ID(msfile, spwID)
    Out='DATA_DESC_ID == %d && SCAN_NUMBER == %d' % (data_desc_id, scanID)
    tb.open(msfile)
    antXantYspw = tb.query(Out, sortlist='TIME')
    timeStamp = np.unique(antXantYspw.getcol('TIME'))
    timeNum = len(timeStamp) 
    uvw = antXantYspw.getcol('UVW')
    tb.close()
    pairNum = int( uvw.shape[1] / timeNum )
    return timeStamp, uvw.reshape(3,timeNum,pairNum)[:,:,xcorr_index].transpose(0,2,1)  # UVW[3, BL, Time]
#
def loadScanSPW(msfile, spwList, scanList): # Load visiblities in all SPWs and scans
    timeStampList, XspecList = [], []
    for spw in spwList:
        XscanList = []
        for scan in scanList:
            timeStamp, Pspec, Xspec = GetVisAllBL(msfile, spw, scan)
            XscanList = XscanList + [Xspec]
            if spw == spwList[0]: timeStampList = timeStampList + [timeStamp]
        XspecList = XspecList + [XscanList]
    return timeStampList, XspecList
#
def GetPSpecScan(msfile, ant, spwID, scanID):
    data_desc_id = SPW2DATA_DESC_ID(msfile, spwID)
    Out='ANTENNA1 == %d && ANTENNA2 == %d && DATA_DESC_ID == %d && SCAN_NUMBER == %d' % (ant, ant, data_desc_id, scanID)
    tb.open(msfile)
    antXantYspw = tb.query(Out)
    colNameList = antXantYspw.colnames()
    colName = 'DATA' 
    if 'FLOAT_DATA' in colNameList: colName = 'FLOAT_DATA'
    timeXY = antXantYspw.getcol('TIME')
    dataXY = antXantYspw.getcol(colName)
    tb.close()
    return timeXY, dataXY.real
# 
def GetPSpec(msfile, ant, spwID):
    data_desc_id = SPW2DATA_DESC_ID(msfile, spwID)
    Out='ANTENNA1 == %d && ANTENNA2 == %d && DATA_DESC_ID == %d' % (ant, ant, data_desc_id)
    tb.open(msfile) 
    antXantYspw = tb.query(Out)
    colNameList = antXantYspw.colnames()
    colName = 'DATA'
    if 'FLOAT_DATA' in colNameList: colName = 'FLOAT_DATA'
    timeXY = antXantYspw.getcol('TIME')
    dataXY = antXantYspw.getcol(colName)
    tb.close()
    return timeXY, dataXY.real
#   
#-------- Time system
def mjd2utc( mjdSec ):        # mjd in [sec]
    utc_dt = MJD_EPOCH + datetime.timedelta(seconds=mjdSec)
    return utc_dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
# 
def AzElMatch( refTime, scanTime, AntID, Az, El ):
    antList = np.unique(AntID).tolist()
    targetAnt = np.argsort([np.ptp(El[np.where(AntID == ant)[0].tolist()]) for ant in antList])[int(len(antList)/2)]
    antTimeIndex = np.where(AntID == targetAnt)[0].tolist()
    time_ptr = [np.argmin(abs(scanTime[antTimeIndex] - now)) for now in refTime]
    return Az[np.array(antTimeIndex)[time_ptr].tolist()], El[np.array(antTimeIndex)[time_ptr].tolist()]
# 
def AzEl2PA(az, el, lat=ALMA_lat): # Azimuth, Elevation, Latitude (default=ALMA) in [rad]
    cos_lat, sin_lat = np.cos(lat), np.sin(lat)
    return np.arctan2( -cos_lat* np.sin(az), (sin_lat* np.cos(el) - cos_lat* np.sin(el)* np.cos(az)) )
#
#-------- Online catalogue
def GetAeff(URI, antMap, band, refMJD):     # Historical aperture efficiency
    antNum = len(antMap)
    Aeff  = np.ones([antNum, 2])
    context = ssl._create_unverified_context()
    response =  urllib.request.urlopen(url = '%sAeB%d.table' % (URI, band), context=context)
    fileLines = response.readlines()
    lineLength = len(fileLines)
    antPolList = fileLines[0].decode('utf-8').split()[1:]
    polList = ['X', 'Y']
    #-------- reference timing
    mjdSec = np.ones(lineLength - 3)
    for line_index, fileLine in enumerate(fileLines[3:]):
        mjdSec[line_index] = qa.convert(fileLine.decode('utf-8').split()[0], 's')['value']
    refpointer = np.argmin(abs(mjdSec - refMJD))
    if (refpointer == 0) | (refpointer == len(mjdSec) - 1) :
        for pol_index, polName in enumerate(polList):
            for ant_index, antName in enumerate(antMap):
                keyhead = antName + '-' + polName
                if keyhead in antPolList:
                    pointer = antPolList.index(keyhead) + 1
                    Aeff[ant_index, pol_index] = float(fileLines[refpointer + 3].decode('utf-8').split()[pointer])
            #
            Aeff[np.where(Aeff[:,pol_index] < 1.001)[0].tolist(),pol_index] = np.median(Aeff[:,pol_index])
        return Aeff
    #
    tmpMJD  = np.array([mjdSec[refpointer-1], mjdSec[refpointer], mjdSec[refpointer+1]])
    for pol_index, polName in enumerate(polList):
        for ant_index, antName in enumerate(antMap):
            keyhead = antName + '-' + polName
            if keyhead in antPolList:
                pointer = antPolList.index(keyhead) + 1
                tmpAeff = np.array([float(fileLines[refpointer+2].decode('utf-8').split()[pointer]), float(fileLines[refpointer+3].decode('utf-8').split()[pointer]), float(fileLines[refpointer+4].decode('utf-8').split()[pointer])])
                Aeff[ant_index, pol_index] = quadratic_interpol(tmpMJD, tmpAeff, refMJD)
        #
        Aeff[np.where(Aeff[:,pol_index] < 1.001)[0].tolist(),pol_index] = np.median(Aeff[:,pol_index])
    #
    return Aeff
#
def GetDterm(URI, antMap, band, refMJD):    # Historical D-term
    print('GetDterm : %sDtermB%d.*.table' % (URI,  band))
    context = ssl._create_unverified_context()
    antNum = len(antMap)
    Dterm  = np.zeros([antNum, 2, 4], dtype=complex)    # Dterm[ant, pol, spw]
    for ant_index in range(antNum):
        ant = antMap[ant_index]
        try:
            response =  urllib.request.urlopen(url = '%sDtermB%d.%s.table' % (URI,  band, ant), context=context)
        except:
            response =  urllib.request.urlopen(url = '%sDtermB%d.%s00.table' % (URI,  band, ant[0:2]), context=context)
        #
        fileLines = response.readlines()
        lineLength = len(fileLines)
        #---- Date
        mjdSec = np.ones(lineLength - 1)
        for line_index in list(range(1, lineLength)):
            mjdSec[line_index - 1] = qa.convert(fileLines[line_index].decode('utf-8').split()[0], 's')['value']
        refpointer = np.argmin(abs(mjdSec - refMJD))
        if (refpointer == 0) | (refpointer == len(mjdSec) - 1) :
            for spwpol_index in list(range(8)):
                pol_index = spwpol_index % 2
                spw_index = spwpol_index // 2
                Dterm[ant_index, pol_index, spw_index] = complex(fileLines[refpointer + 1].decode('utf-8').split()[spwpol_index + 1].replace('i','j'))
            #
            return Dterm
        #
        tmpMJD  = np.array([mjdSec[refpointer-1], mjdSec[refpointer], mjdSec[refpointer+1]])
        for spwpol_index in list(range(8)):
            pol_index = spwpol_index % 2
            spw_index = spwpol_index // 2
            tmpD = np.array([ complex(fileLines[refpointer].decode('utf-8').replace('i', 'j').split()[spwpol_index + 1]), complex(fileLines[refpointer+1].decode('utf-8').replace('i', 'j').split()[spwpol_index + 1]), complex(fileLines[refpointer+2].decode('utf-8').replace('i', 'j').split()[spwpol_index + 1])])
            Dterm[ant_index, pol_index, spw_index] = quadratic_interpol(tmpMJD, tmpD.real, refMJD) + (0.0 + 1.0j)*quadratic_interpol(tmpMJD, tmpD.imag, refMJD)
        #
    #
    return Dterm
#
def GetSourceDic(msfile):                   # source Dictionary
    from Grid import sourceRename
    import json
    msmd.open(msfile)
    tb.open(msfile + '/FIELD')
    fieldList = msmd.fieldnames(); fieldID = set()
    mjdSec = tb.getcol('TIME')[0]   
    #---- Solar Position to calculate sun separation angle
    base_url = "https://ssd.jpl.nasa.gov/api/horizons.api"
    JPLparams = {           # Query for JPL Horizons
        "format": "json",
        "COMMAND": "'10'",  # Sun
        "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'OBSERVER'",
        "CENTER": "'500'",  # Geocenter
        "START_TIME": f"'{mjd2utc(mjdSec).split('.')[0].replace('T',' ')}'",
        "STOP_TIME":  f"'{mjd2utc(mjdSec+60).split('.')[0].replace('T',' ')}'",
        "STEP_SIZE": "'1 min'",
        "QUANTITIES": "'2'",
        "ANG_FORMAT": "'DEG'"}
    url_parts = urllib.parse.urlencode(JPLparams, quote_via=urllib.parse.quote)
    request_url = f"{base_url}?{url_parts}"
    ssl_context = ssl._create_unverified_context()
    req = urllib.request.Request( request_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, context=ssl_context) as response:
        response_data = response.read().decode("utf-8")
    data = json.loads(response_data)
    result_text = data.get("result", "")
    data_match = re.search(r"\$\$SOE\n(.*)\n\$\$EOE", result_text, re.DOTALL)
    data_line = data_match.group(1).strip()
    float_values = re.findall(r"-?\d+\.\d+", data_line)
    sunPos = np.array([float(float_values[0]), float(float_values[1])]) / RADDEG
    for field in fieldList: fieldID = fieldID | set(msmd.fieldsforname(field))
    fieldID = list(fieldID); fieldID.sort()
    fieldDic = dict(zip(fieldID, [[]]* len(fieldID)))
    fieldPos  = tb.getcol('PHASE_DIR')[:,0].T
    fieldList = sourceRename(fieldList)
    for ID in fieldDic.keys():
        deltaRA = fieldPos[ID,0] - sunPos[0]
        deltaDC = fieldPos[ID,1] - sunPos[1]
        fieldDic[ID] = {
            'Name': fieldList[ID],
            'RA'  : fieldPos[ID,0],
            'DEC' : fieldPos[ID,1],
            'SA'  : RADDEG*2* np.arcsin(np.sqrt(np.sin(0.5* deltaDC)**2 + np.cos(fieldPos[ID,1])* np.cos(sunPos[1]) * np.sin(0.5*deltaRA)**2))}
    msmd.close()
    tb.close()
    return fieldDic
#
def GetVisAllBL(msfile, spwID, scanID):         # visiblities of wll baselines
    Antenna1, Antenna2 = GetBaselineIndex(msfile, spwID, scanID)
    acorr_index, xcorr_index = corrAddress(Antenna1, Antenna2)
    #
    data_desc_id = SPW2DATA_DESC_ID(msfile, spwID)
    Out='DATA_DESC_ID == %d && SCAN_NUMBER == %d' % (data_desc_id, scanID)
    tb.open(msfile)
    antXantYspw = tb.query(Out, sortlist='TIME')
    timeStamp = np.unique(antXantYspw.getcol('TIME'))
    timeNum = len(timeStamp)
    colName = 'DATA'
    colNameList = antXantYspw.colnames()    
    if 'FLOAT_DATA' in colNameList: colName = 'FLOAT_DATA'
    dataXY = antXantYspw.getcol(colName) 
    tb.close()
    corrNum, chNum, pairNum = dataXY.shape[0], dataXY.shape[1], int( dataXY.shape[2] / timeNum )
    dataXY = dataXY.reshape(corrNum, chNum, timeNum, pairNum).transpose(0,1,3,2)
    return timeStamp, dataXY[:,:,acorr_index], dataXY[:,:,xcorr_index]  # AC[POL, CH, ANT, Time], XC[POL, CH, BL, Time]
# 
#-------- Algebra
def quadratic_interpol(x, y, t):    # quadratic interpolation 
    # x[3]  : periodic arguments
    # y[3]  : values at x
    # t     : to output y[t] 
    moment = y[0] - 2.0*y[1] + y[2]
    refPhase  = (t - x[1]) / np.mean(np.diff(x))
    return 0.5* refPhase* (refPhase* moment + y[2] - y[0]) + y[1]
#
def linearRegression( x, y, err):
    weight = 1.0 / err**2
    Sw, Swxx, Swx, Swy, Swxy = np.sum(weight), np.sum( weight* x**2), weight.dot(x), weight.dot(y), np.sum(weight* x* y)
    det    = Sw* Swxx - Swx**2
    sol    = np.array([Swxx* Swy - Swx* Swxy, Sw* Swxy - Swx* Swy]) / det
    solerr = np.sqrt(np.array([Swxx, Sw]) / det)
    return sol, solerr
#
def AllanVarPhase(phase, lag):  # phase in radian
    vecSize = len(phase);   diffSize = vecSize - lag;   avSize = diffSize - lag
    x = np.exp((0.0 + 1.0j)* phase)
    temp = x[lag:vecSize] * x[0:diffSize].conjugate()
    temp2= temp[lag:diffSize] * temp[0:avSize].conjugate()
    return np.angle(temp2).dot(np.angle(temp2)) / (2* avSize* lag* lag)
#
def bunchVec( spec, bunchNum ):     # channel binning
    if bunchNum == 1:
        return spec
    else: 
        chNum = int(len(spec)/bunchNum)
        totalNum = int(chNum* bunchNum)
        return(np.mean( spec[0:totalNum].reshape(chNum, bunchNum), axis=1))
#
def smoothValue( refTime, refValue, givenTime ):     # Return value at given time
    SP  = UnivariateSpline( refTime, refValue, s=0.25*np.std(refValue))
    return SP(givenTime)
#
def ATAmatrix(Gain):  # Gain is a vector of antenna-based gain amplitude (real)
    antNum = len(Gain); normG = Gain.dot(Gain)
    PTP = np.zeros([antNum, antNum]) + Gain
    for ant_index in range(antNum):
        PTP[ant_index,:] *= Gain[ant_index]
        PTP[ant_index, ant_index] = normG - Gain[ant_index]**2
    #
    return PTP
#   
def CTCmatrix(Gain):  # Gain is a vector of antenna-based gain amplitude (imag)
    antNum = len(Gain); normG = Gain.dot(Gain)
    PTP = np.zeros([antNum, antNum]) + Gain
    for ant_index in range(antNum):
        PTP[ant_index,:] *= (-Gain[ant_index])
        PTP[ant_index, ant_index] = normG - Gain[ant_index]**2
    #
    return PTP
#   
def ATBmatrix(Gain):  # Gain is a vector of antenna-based complex gain
    antNum = len(Gain); normG = Gain.real.dot(Gain.imag)
    PTP = np.zeros([antNum, antNum]) + Gain.imag
    for ant_index in range(antNum): 
        PTP[ant_index,:] = Gain.real[ant_index]* Gain.imag + Gain.imag[ant_index]* Gain.real
        PTP[ant_index, ant_index] = 0.0
    #
    return PTP[1:antNum]
#   
def PMatrix(CompSol):
    antNum = len(CompSol); matSize = 2*antNum-1
    PM = np.zeros([matSize, matSize])
    PM[0:antNum][:,0:antNum]   = ATAmatrix(CompSol.real) + CTCmatrix(CompSol.imag)
    PM[antNum:matSize][:,antNum:matSize] = ATAmatrix(CompSol.imag)[1:antNum][:,1:antNum] + CTCmatrix(CompSol.real)[1:antNum][:,1:antNum]
    PM[antNum:matSize][:,0:antNum]   = ATBmatrix(CompSol)
    PM[0:antNum][:,antNum:matSize]   = PM[antNum:matSize][:,0:antNum].T
    return PM
# 
def PTdotR(CompSol, Cresid):
    antNum = len(CompSol)
    blNum = int(antNum* (antNum-1) / 2)
    ant0, ant1= np.array(ANT0[0:blNum]), np.array(ANT1[0:blNum])
    PTR = np.zeros(2*antNum)
    for ant_index in range(antNum):
        index0 = np.where(ant0 == ant_index)[0].tolist()
        index1 = np.where(ant1 == ant_index)[0].tolist()
        PTR[list(range(ant_index))]          += (CompSol[ant_index].real* Cresid[index0].real + CompSol[ant_index].imag* Cresid[index0].imag)
        PTR[list(range(ant_index+1,antNum))] += (CompSol[ant_index].real* Cresid[index1].real - CompSol[ant_index].imag* Cresid[index1].imag)
        PTR[list(range(antNum, antNum+ant_index))]    += (CompSol[ant_index].imag* Cresid[index0].real - CompSol[ant_index].real* Cresid[index0].imag)
        PTR[list(range(antNum+ant_index+1,2*antNum))] += (CompSol[ant_index].imag* Cresid[index1].real + CompSol[ant_index].real* Cresid[index1].imag)
    # 
    return PTR[list(range(antNum)) + list(range(antNum+1, 2*antNum))]
#  
def InvMullerMatrix(Dx0, Dy0, Dx1, Dy1):
    return np.array([     
        [1.0, -Dx1.conjugate(), -Dx0, Dx0* Dx1.conjugate()],
        [-Dy1.conjugate(), 1.0, Dx0* Dy1.conjugate(), -Dx0],
        [-Dy0, Dy0* Dx1.conjugate(), 1.0, -Dx1.conjugate()],
        [Dy0* Dy1.conjugate(), -Dy0, -Dy1.conjugate(), 1.0]]) / ((1.0 - Dx0* Dy0)*(1.0 - Dx1.conjugate()* Dy1.conjugate()))
#
def InvPAVector(PA, Unity):
    cs, sn = np.cos(2.0* PA), np.sin(2.0* PA)
    Zeroty = 0.0* Unity
    return 0.5*np.array([
        [Unity, Zeroty, Zeroty, Unity],
        [ cs, -sn, -sn, -cs],
        [ sn,  cs,  cs, -sn],
        [Zeroty,-1.0j*Unity,1.0j*Unity, Zeroty]])
#
#-------- Calibration
def beamF(disk2FWHMratio):     # diskR / FWHM ratio
    disk2sigma = disk2FWHMratio * 2.3548200450309493   # FWHM / (2.0* np.sqrt(2.0* log(2.0)))
    return( 2.0* (1.0 - np.exp(-0.5* (disk2sigma)**2)) / (disk2sigma**2) )
#
def gainComplexVec( bl_vis, niter=2 ):      # Solve antenna-based gains using visibilities
    # bl_vis[baseline, channel]
    ChavVis = np.median(bl_vis.real, axis=1) + (0.0+1.0j)*np.median(bl_vis.imag, axis=1)
    #ChavVis = np.mean(bl_vis.real, axis=1) + (0.0+1.0j)*np.mean(bl_vis.imag, axis=1)
    blNum, chNum  =  bl_vis.shape[0], bl_vis.shape[1]
    antNum =  Bl2Ant(blNum)[0]
    ant0, ant1, kernelBL = ANT0[0:blNum], ANT1[0:blNum], KERNEL_BL[range(antNum-1)].tolist()
    #---- Initial solution
    CompSol = np.append(np.sqrt(abs(ChavVis[0])) + 0j, ChavVis[kernelBL])
    CompSol[1:antNum] /=  CompSol[0]
    #---- Global iteration
    PTP = PMatrix(CompSol)
    L = np.linalg.cholesky(PTP)
    def GlobalGainSolve(vis):
        Cresid = vis - CompSol[ant0]* CompSol[ant1].conjugate()
        t = np.linalg.solve(L, PTdotR(CompSol, Cresid))
        correction = np.linalg.solve(L.T, t)
        return CompSol + correction[list(range(antNum))] + 1.0j* np.append(0, correction[list(range(antNum, 2*antNum-1))])
    Solution = np.apply_along_axis(GlobalGainSolve, 0, bl_vis)
    #---- Local iteration
    def LocalGainSolve(visSol):
        vis, sol = visSol[0:blNum], visSol[blNum:blNum+antNum]
        PTP = PMatrix(sol)
        L = np.linalg.cholesky(PTP)
        Cresid = vis - sol[ant0]* sol[ant1].conjugate()
        t = np.linalg.solve(L, PTdotR(sol, Cresid))
        correction = np.linalg.solve(L.T, t)
        return sol + correction[list(range(antNum))] + 1.0j* np.append(0, correction[list(range(antNum, 2*antNum-1))])
    for iter_index in list(range(niter)): Solution = np.apply_along_axis(LocalGainSolve, 0, np.concatenate([bl_vis, Solution]))
    return Solution
#
def gainComplex( bl_vis, niter=4 ):         # Solve antenna-based gains using visibilities
    blNum  =  len(bl_vis)
    antNum =  Bl2Ant(blNum)[0]
    ant0, ant1, kernelBL = ANT0[0:blNum], ANT1[0:blNum], KERNEL_BL[range(antNum-1)].tolist()
    CompSol = np.zeros(antNum, dtype=complex)
    #---- Initial solution
    CompSol[0] = 1.0e-6 + np.sqrt(abs(bl_vis[0])) + 0j     # add 1.0e-6 to avoid division by 0
    CompSol[1:antNum] = bl_vis[kernelBL] / CompSol[0]
    #----  Iteration
    for iter_index in list(range(niter)):
        PTP        = PMatrix(CompSol)
        L          = np.linalg.cholesky(PTP)         # Cholesky decomposition
        Cresid     = bl_vis - CompSol[ant0]* CompSol[ant1].conjugate()
        t          = np.linalg.solve(L, PTdotR(CompSol, Cresid))
        correction = np.linalg.solve(L.T, t)
        CompSol    = CompSol + correction[list(range(antNum))] + 1.0j* np.append(0, correction[list(range(antNum, 2*antNum-1))])
    return CompSol
#
def spec2corr(spec):
    nspec = len(spec)     
    tmpspec = np.append(spec, np.zeros(nspec, dtype=complex))
    corr = scipy.fft.ifft(tmpspec)
    return np.append(corr[nspec:(2*nspec)], corr[:nspec])
#
def delay_cal( spec, delay ):
    # spec : input spectrum (complex)
    # delay : delay[1] = initial phase, delay[2] = delay
    # delay_cal() returns delay-calibrated spectrum
    #
    nspec = len( spec )
    twiddle = np.exp( np.pi* delay* np.multiply(list(range(-nspec, nspec, 2)), 0.5j) / nspec )
    return np.multiply(spec, twiddle)
#
def delay_search( spec ):                   # Find delay residual
    nspec = len( spec ) 
    #-------- Search for delay
    corrAbs = abs(spec2corr(spec))
    if( max(corrAbs) == 0.0 ): return 0.0, 1.0e-20
    delay = np.argmax(corrAbs) - nspec # Coarse Delay
    trial_delay = (delay + np.multiply(range(-2,3), 0.5)).tolist()
    trial_amp = np.array([abs(np.mean(delay_cal(spec, temporal_delay))) for temporal_delay in trial_delay])
    fit = np.polyfit(trial_delay, trial_amp, 2)
    bestDelay = -fit[1]/(2.0*fit[0])
    residualPhase = np.angle( delay_cal(spec, bestDelay) )
    snr = 4 / (AllanVarPhase(residualPhase, int(nspec/4))* nspec* np.sqrt(nspec) + 1.0e-9)
    return bestDelay, snr
#
def SPWalign(spwGain):       # spwGain[spw, pol, ant, time]
    timeNum  = spwGain.shape[3]
    spwTwiddle= np.mean(spwGain, axis=3).transpose(2,1,0) # [ant, pol, spw]
    for ant_index in list(range(spwTwiddle.shape[0])):
        refGain = np.ones(timeNum, dtype=complex)
        gainOffset = spwGain[:,:,ant_index].dot(refGain)
        refGain = np.mean(spwGain[:,:,ant_index].transpose(2,0,1)* gainOffset.conjugate(), axis=(1,2))
        gainOffset = spwGain[:,:,ant_index].dot(refGain)
        spwTwiddle[ant_index] = (gainOffset / abs(gainOffset)).T
    return spwTwiddle
#
def CrossPolBL(Xspec, blInv):
    Neg, Pos = (0.0 + np.array(blInv)), (1.0 - np.array(blInv))
    Tspec = Xspec.copy()
    Tspec[0]   = (Xspec[0].transpose(0,2,1)* Pos + Xspec[0].conjugate().transpose(0,2,1)* Neg).transpose(0,2,1) # XX
    Tspec[1]   = (Xspec[1].transpose(0,2,1)* Pos + Xspec[2].conjugate().transpose(0,2,1)* Neg).transpose(0,2,1) # XY
    Tspec[2]   = (Xspec[2].transpose(0,2,1)* Pos + Xspec[1].conjugate().transpose(0,2,1)* Neg).transpose(0,2,1) # YX
    Tspec[3]   = (Xspec[3].transpose(0,2,1)* Pos + Xspec[3].conjugate().transpose(0,2,1)* Neg).transpose(0,2,1) # YY
    return Tspec
# 
def CrossPolBP(Xspec):  # full-polarization bandpass: Xspec[pol, ch, bl, time]
    polNum, chNum, blNum, timeNum = Xspec.shape
    chRange = list(range(int(0.1*chNum), int(0.9*chNum)))
    antNum = Bl2Ant(blNum)[0]
    ant0, ant1 = ANT0[0:blNum], ANT1[0:blNum]
    polXindex, polYindex = (np.arange(4)//2).tolist(), (np.arange(4)%2).tolist()
    BP_ant  = np.ones([antNum, 2, chNum], dtype=complex)
    #---- Gain Cal for coherent averating
    Gain = np.array([gainComplexVec(np.mean(Xspec[0,chRange], axis=0)), gainComplexVec(np.mean(Xspec[3,chRange], axis=0))])
    XPspec = np.mean((abs(Gain[polYindex][:,ant0]* Gain[polXindex][:,ant1])* Xspec.transpose(1,0,2,3) / (Gain[polYindex][:,ant0]* Gain[polXindex][:,ant1].conjugate())).transpose(1,0,2,3), axis=3)
    #---- Tentative BP
    BP_ant[:,0], BP_ant[:,1] = gainComplexVec(XPspec[0].T), gainComplexVec(XPspec[3].T)
    XPspec = np.mean((abs(Gain[polYindex][:,ant0]* Gain[polXindex][:,ant1])* Xspec.transpose(1,0,2,3) / (Gain[polYindex][:,ant0]* Gain[polXindex][:,ant1].conjugate())).transpose(1,0,2,3), axis=3)
    #---- improved BP
    BP_ant[:,0], BP_ant[:,1] = gainComplexVec(XPspec[0].T), gainComplexVec(XPspec[3].T)
    BPCaledXspec = XPspec.transpose(2, 0, 1)* abs(BP_ant[ant0][:,polYindex]* BP_ant[ant1][:,polXindex])**2 /(BP_ant[ant0][:,polYindex]* BP_ant[ant1][:,polXindex].conjugate())
    del XPspec
    #---- Amplitude normalization
    for pol_index in [0,1]:
        ant_index = np.where( abs(np.mean(BP_ant[:,pol_index][:,chRange], axis=1)) > 0.1* np.median( abs(np.mean(BP_ant[:,pol_index][:,chRange], axis=1)) ))[0].tolist()
        BP_ant[ant_index, pol_index] = (BP_ant[ant_index, pol_index].T / np.mean(abs(BP_ant[ant_index, pol_index][:,chRange]), axis=1)).T
    BPCaledXYSpec = np.mean(BPCaledXspec[:,1], axis=0) +  np.mean(BPCaledXspec[:,2], axis=0).conjugate()
    XYdelay, XYsnr = delay_search( BPCaledXYSpec[chRange] )
    XYdelay = (float(chNum) / float(len(chRange)))* XYdelay
    return BP_ant, BPCaledXYSpec, XYdelay, Gain, XYsnr
#
def VisMuiti_solveD(Vis, QCpUS, UCmQS, Dx=[], Dy=[], I=1.0):        # Solve for D-term
    # Vis (input) : full-polarization visibilities [pol, bl, PA]
    # QCpUS  (input) : Q cos(2PA) + U sin(2PA)
    # UCmQS  (input) : U cos(2PA) - Q sin(2PA)
    PAnum, blNum = len(QCpUS), Vis.shape[1]; antNum = Bl2Ant(blNum)[0]
    ant0, ant1 = np.array(ANT0[0:blNum]), np.array(ANT1[0:blNum])
    #-------- Residual Vector
    residXX, residXY, residYX, residYY = Vis[0] - (I + QCpUS), Vis[1] - UCmQS, Vis[2] - UCmQS, Vis[3] - (I - QCpUS)
    ssqUCmQS = UCmQS.dot(UCmQS)     # sum( UCmQS^2 )
    ImmQCpUS = (I - QCpUS).dot(I - QCpUS)  # sum (I - QCpUS)^2)
    IppQCpUS = (I + QCpUS).dot(I + QCpUS)  # sum (I + QCpUS)^2)
    IpmQCpUS = (I + QCpUS).dot(I - QCpUS)  # sum (I + QCpUS)(I - QCpUS)
    if len(Dx) == 0 :                    # Start over the initial D-term value
        #-------- <XX*> to determine Dx (initial value)
        PTP_inv = np.zeros([2*antNum-1, 2*antNum-1])
        PTP_inv[0:antNum][:,0:antNum] = ((2.0* antNum - 2.0)*np.identity(antNum) - 1.0) / (2.0* (antNum - 2.0)* (antNum - 3.0))
        PTP_inv[antNum:(2*antNum-1)][:,antNum:(2*antNum-1)] = (np.identity(antNum - 1) + 1.0) / antNum
        PTP_inv *= (blNum/ ssqUCmQS)
        PTY = np.zeros([2*antNum-1])
        resid = Vis[0] - (I + QCpUS)
        for ant_index in range(1, antNum):
            index0, index1 = np.where(ant0 == ant_index)[0].tolist(), np.where(ant1 == ant_index)[0].tolist()
            PTY[ant_index] = np.sum((residXX[index0].real).dot(QCpUS)) + np.sum((residXX[index1].real).dot(QCpUS))
            PTY[antNum + ant_index - 1] = np.sum((residXX[index0].imag).dot(QCpUS)) - np.sum((residXX[index1].imag).dot(QCpUS))
        #
        index1 = np.where(ant1 == 0)[0].tolist(); PTY[0] = np.sum((residXX[index1].real).dot(QCpUS))
        Solution = PTP_inv.dot(PTY); newDx = Solution[0:antNum] + (1.0j)* np.append(0.0, Solution[antNum:(2*antNum-1)])
        #-------- <YY*> to determine Dy (initial value)
        for ant_index in range(antNum):
            index0, index1 = np.where(ant0 == ant_index)[0].tolist(), np.where(ant1 == ant_index)[0].tolist()
            PTY[ant_index] = np.sum((residYY[index0].real).dot(QCpUS)) + np.sum((residYY[index1].real).dot(QCpUS))
            PTY[antNum + ant_index - 1] = np.sum((residYY[index0].imag).dot(QCpUS)) - np.sum((residYY[index1].imag).dot(QCpUS))
        #
        index1 = np.where(ant1 == 0)[0].tolist(); PTY[0] = np.sum((residYY[index1].real).dot(QCpUS))
        Solution = PTP_inv.dot(PTY); newDy = Solution[0:antNum] + (1.0j)* np.append(0.0, Solution[antNum:(2*antNum-1)])
    else:
        newDx, newDy = Dx.copy(), Dy.copy()
    #
    #-------- <XY*> and <YX*> to determine Dx and Dy
    PTP = np.zeros([4*antNum, 4*antNum])            # (Dx.real, Dx.imag, Dy.real, Dy.imag)
    PTP[0:2*antNum][:,0:2*antNum] = ImmQCpUS* (antNum - 1.0)* np.identity(2*antNum)    # P00, P01, P10, and P11
    PTP[2*antNum:4*antNum][:,2*antNum:4*antNum] = IppQCpUS* (antNum - 1.0)* np.identity(2*antNum)  # P22, P23, P32, and P33
    PTP[2*antNum:3*antNum][:,0:antNum] = IpmQCpUS* (1.0 - np.identity(antNum))         # P02
    PTP[3*antNum:4*antNum][:,antNum:2*antNum] = -PTP[2*antNum:3*antNum][:,0:antNum]    # P13
    PTP[0:antNum][:,2*antNum:3*antNum] = PTP[2*antNum:3*antNum][:,0:antNum]            # P20
    PTP[antNum:2*antNum][:,3*antNum:4*antNum] = -PTP[2*antNum:3*antNum][:,0:antNum]    # P31
    PTY = np.zeros(4* antNum)
    for ant_index in list(range(antNum)):
        index0, index1 = np.where(ant0 == ant_index)[0].tolist(), np.where(ant1 == ant_index)[0].tolist()
        residXY[index0] -= newDx[ant_index]* (I - QCpUS)
        residXY[index1] -= newDy[ant_index].conjugate()* (I + QCpUS)
        residYX[index0] -= newDy[ant_index]* (I + QCpUS)
        residYX[index1] -= newDx[ant_index].conjugate()* (I - QCpUS)
    #
    #-------- PTR vector
    for ant_index in list(range(antNum)):
        index0, index1 = np.where(ant0 == ant_index)[0].tolist(), np.where(ant1 == ant_index)[0].tolist()
        PTY[ant_index] += (I - QCpUS).dot(np.sum(residXY[index0].real, axis=0))
        PTY[ant_index] += (I - QCpUS).dot(np.sum(residYX[index1].real, axis=0))
        PTY[antNum + ant_index] += (I - QCpUS).dot(np.sum(residXY[index0].imag, axis=0))
        PTY[antNum + ant_index] -= (I - QCpUS).dot(np.sum(residYX[index1].imag, axis=0))
        PTY[2*antNum + ant_index] += (I + QCpUS).dot(np.sum(residXY[index1].real, axis=0))
        PTY[2*antNum + ant_index] += (I + QCpUS).dot(np.sum(residYX[index0].real, axis=0))
        PTY[3*antNum + ant_index] -= (I + QCpUS).dot(np.sum(residXY[index1].imag, axis=0))
        PTY[3*antNum + ant_index] += (I + QCpUS).dot(np.sum(residYX[index0].imag, axis=0))
    #
    #-------- Solution
    L = np.linalg.cholesky(PTP)
    t = np.linalg.solve(L, PTY)
    Solution = np.linalg.solve(L.T, t)
    #-------- Correction
    newDx += Solution[0:antNum] + (1.0j)* Solution[antNum:2*antNum]
    newDy += Solution[2*antNum:3*antNum] + (1.0j)* Solution[3*antNum:4*antNum]
    return newDx, newDy
#

