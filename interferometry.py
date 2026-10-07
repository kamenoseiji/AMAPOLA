import numpy as np
#-------- Constants
Tcmb = 2.725        # CMB temperature
kb   = 1.38064852e3 # Boltzman constant (* 1e26 for Jy scaling)
BANDPA = [0.0, 45.0, -45.0, 80.0, -80.0, 45.0, -45.0, 36.45, 90.0, -90.0, 0.0]   # X-pol orientation for Band-1, 2, 3, 4, 5, 6, 7, 8, 9, and 10
BANDFQ = [0.0, 43.2, 75.0, 97.5, 132.0, 183.0, 233.0, 343.5, 460.0, 650.0, 870.0]   # Standard frequency [GHz]
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
def GetAntD(antList):                       # Antenna diameter list
    antD = 12.0* np.ones(len(antList))
    for ant_index, antName in enumerate(antList):
        if antName.find('C') > -1: antD[ant_index] = 7.0
    return antD
# 
def GetBandNames(msfile, atmSPWs=[]):       # Receiver band name
    if len(atmSPWs) < 1: atmSPWs = GetAtmSPWs(msfile)
    msmd.open(msfile)
    atmspwNames = msmd.namesforspws(atmSPWs)
    atmBandNames, atmPattern = [], r'RB_..'
    for spwName in atmspwNames : atmBandNames = atmBandNames + re.findall(atmPattern, spwName)
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
#-------- Indexing
def indexList( refArray, motherArray ):     # Compare two arrays and return matched index
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
#-------- Algebra
def quadratic_interpol(x, y, t):    # quadratic interpolation 
    # x[3]  : periodic arguments
    # y[3]  : values at x
    # t     : to output y[t] 
    moment = y[0] - 2.0*y[1] + y[2]
    refPhase  = (t - x[1]) / np.mean(np.diff(x))
    return 0.5* refPhase* (refPhase* moment + y[2] - y[0]) + y[1]
#


