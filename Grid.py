import os
import math
import numpy as np
import scipy
#SSOCatalog = ['Uranus', 'Neptune', 'Callisto', 'Ganymede', 'Titan', 'Io', 'Europa', 'Ceres', 'Pallas', 'Vesta', 'Juno', 'Mars', 'Mercury', 'Venus']
SSOCatalog = ['Uranus', 'Neptune', 'Callisto', 'Ganymede', 'Titan', 'Io', 'Europa', 'Ceres', 'Pallas', 'Vesta', 'Juno', 'Mars', 'Mercury', 'Venus']
SSOscore   = [
[ 5.0,     4.0,       1.0,        1.0,        0.1,     0.2,  0.3,      0.2,     0.1,     0.1,    10.0,   10.0,     10.0],   # Band 1
[ 6.0,     5.0,       1.0,        1.0,        0.1,     0.6,  0.5,      0.3,     0.2,     0.2,    10.0,   10.0,     10.0],   # Band 2
[ 7.0,     6.0,       1.0,        1.0,        0.2,     0.7,  0.7,      0.5,     0.3,     0.3,    10.0,   10.0,     10.0],   # Band 3
[ 8.0,     7.0,       2.0,        2.0,        0.3,     0.9,  0.9,      0.6,     0.4,     0.4,     8.0,   10.0,      8.0],   # Band 4
[ 9.0,     8.0,       3.0,        3.0,        0.5,     1.0,  1.0,      0.8,     0.5,     0.5,     4.0,    4.0,      4.0],   # Band 5
[10.0,     9.0,       4.0,        4.0,        5.0,     3.0,  3.0,      1.0,     0.6,     0.6,     2.0,    2.0,      2.0],   # Band 6
[10.0,     9.0,       5.0,        5.0,        7.0,     4.0,  4.0,      3.0,     0.7,     0.7,     1.0,    1.0,      1.0],   # Band 7
[ 5.0,     6.0,       7.0,        7.0,        7.0,     8.0,  4.0,      3.0,     0.7,     0.7,     1.0,    1.0,      1.0],   # Band 8
[ 2.0,     3.0,       7.0,        7.0,        7.0,     9.0,  4.0,     10.0,     7.0,     4.0,     1.0,    1.0,      1.0]]   # Band 9
ELshadow = math.pi* 40.0 / 180.0    # EL to avoid refscan
SunAngleThresh = 5.0
#-- Acceptable opacity       B3   B4    B5   B6   B7   B8,  B9, B10
TauLimit = [0.0,  0.1, 0.1, 0.2, 0.2, 0.25, 0.5, 0.5, 0.5, 0.5, 0.6]
#-- Source name alias
sourceDic = {
'J0006-063':'J0006-0623',
'0006-063':'J0006-0623',
'J0237+288':'J0237+2848',
'NGC_1052': 'J0241-0815',
'3c84':'J0319+4130',
'0334-401':'J0334-4008',
'J0334-401':'J0334-4008',
'J0423-013':'J0423-0120',
'J0510+180':'J0510+1800',
'J0519-454':'J0519-4546',
'J0522-364':'J0522-3627',
'J0538-440':'J0538-4405',
'J0750+125':'J0750+1231',
'J0854+201':'J0854+2006',
'J1037-295':'J1037-2934',
'J1058+015':'J1058+0133',
'J1107-448':'J1107-4449',
'J1146+399':'J1146+3958',
'3c273':'J1229+0203',
'3C273':'J1229+0203',
'3c279':'J1256-0547',
'3C279':'J1256-0547',
'3C_279':'J1256-0547',
'J1337-129':'J1337-1257',
'1337-129':'J1337-1257',
'3C286':'J1331+3030',
'3C_286':'J1331+3030',
'1427-421':'J1427-4206',
'J1427-421':'J1427-4206',
'J1517-243':'J1517-2422',
'1517-243':'J1517-2422',
'J1550+054':'J1550+0527',
'1550+054':'J1550+0527',
'J1613-586':'J1617-5848',
'1613-586':'J1617-5848',
'J1625-254':'J1625-2527',
'3c345':'J1642+3948',
'J1733-130':'J1733-1304',
'1733-130':'J1733-1304',
'nrao530':'J1733-1304',
'J1751+096':'J1751+0939',
'1751+096':'J1751+0939',
'J1924-292':'J1924-2914',
'1924-292':'J1924-2914',
'J2025+337':'J2025+3343',
'J2056-472':'J2056-4714',
'2056-472':'J2056-4714',
'2157-694':'J2157-6941',
'J2148+069':'J2148+0657',
'J2232+117':'J2232+1143',
'3c454.3':'J2253+1608',
'J2258-279':'J2258-2758',
'BL_Lac':'J2202+4216',
'VLA_J1650+0824':'J1650+0824',
'VLA_J1652+0618':'J1652+0618',
'VLA_J1656+1826':'J1656+1826',
'VLA_J1658+0741':'J1658+0741',
'VLA_J1658+0515':'J1658+0515',
'VLA_J1700+0522':'J1700+0522',
'VLA_J1707+0148':'J1707+0148',
'VLA_J1707+1331':'J1707+1331',
'VLA_J1707+1846':'J1707+1846',
'VLA_J1708+0035':'J1708+0035',
'VLA_J1716+2152':'J1716+2152',
'VLA_J1719+0658':'J1719+0658',
'VLA_J1719+1745':'J1719+1745',
'VLA_J1719+0817':'J1719+0817',
'VLA_J1722+1013':'J1722+1013',
'VLA_J1726+0639':'J1726+0639',
'VLA_J1728+1215':'J1728+1215',
'VLA_J1728+0427':'J1728+0427',
'VLA_J1730+0024':'J1730+0024',
'VLA_J1734+0926':'J1734+0926',
'VLA_J1737+0621':'J1737+0621',
'VLA_J1740+2211':'J1740+2211',
'VLA_J1730+0024':'J1730+0024',
'VLA_J1734+0926':'J1734+0926',
'VLA_J1737+0621':'J1737+0621',
'VLA_J1740+2211':'J1740+2211'}

def sourceRename(sourceList):
    renameList = []
    for srcname in sourceList:
        srcname = srcname.replace('*','')
        renameList = renameList + [sourceDic.get(srcname, srcname)]
    return renameList
#
def AeNominal(msfile, antList):
    msmd.open(msfile)
    antDia = [msmd.antennadiameter(antName)['value'] for antName in antList]
    msmd.close()
    msmd.done()
    return 0.72* 0.25* np.pi* antDia**2      # Nominal Collecting Area
def ArrayAltAz(msfile):            # Antenna position in Alt-Az coordinates relative to the array center (EW, NS, H)
    from interferometry import GetAntName, GetAntPos, GetAntD
    antList = GetAntName(msfile)
    antPos  = GetAntPos(msfile)
    antDia  = GetAntD(antList)
    antDic = dict(zip(antList, [[]]* len(antList)))
    refVector  = np.mean(antPos, axis=1)                        # Array reference position
    refVector_n = refVector/np.sqrt(refVector.dot(refVector))   # Normal vector 
    antRelPos = (antPos.T - refVector).T
    #---- (X,Y,Z) -> (EW, NS, H)
    R_xy, R_z = np.sqrt(refVector_n[:2].dot(refVector_n[:2])), refVector_n[2]
    CS, SN = refVector_n[0]/R_xy, refVector_n[1]/R_xy
    Rm = np.array([ [R_z*CS, R_z*SN, -R_xy], [-SN,  CS, 0.0], [R_xy*CS, R_xy*SN, R_z]])    # Rotation matrix
    ENpos = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]).dot(Rm.dot(antRelPos)) # East-West, North-South, Height
    for ant_index, ant in enumerate(antDic.keys()): antDic[ant] = {'EN' : ENpos[:,ant_index], 'Dia' : antDia[ant_index] }
    return antDic
def AntShadow(antDic, az, el):        # Shadowing at az, el (rad)
    directionVector = np.array([np.cos(el)*np.sin(az), np.cos(el)*np.cos(az), np.sin(el)])
    antList = list(antDic.keys())
    for ant in antDic.keys():
        nearAntList = [otherAnt for otherAnt in antList if (otherAnt != ant) & ((antDic[otherAnt]['EN'] - antDic[ant]['EN']).dot(antDic[otherAnt]['EN'] - antDic[ant]['EN']) < (antDic[otherAnt]['Dia']* (1 + 1/np.tan(el)))**2)]
        if len(nearAntList) < 1: antDic[ant]['Shadow'] = 0.0; continue
        blVec = [(antDic[otherAnt]['EN'] - antDic[ant]['EN']) for otherAnt in nearAntList]
        frontA= [blVec[other_index].dot(directionVector) for other_index, otherAnt in enumerate(nearAntList)]   # positive -> front side (shadow), negative -> backside (not shadow)
        projBL= [blVec[other_index] - frontA[other_index]*directionVector for other_index, otherAnt in enumerate(nearAntList)]  # Projected baseline vector from the source
        projSep = [np.sqrt(projBL[other_index].dot(projBL[other_index])) for other_index, otherAnt in enumerate(nearAntList)]   # Projected baseline lengthf
        shadowFrac = [(0.5*(antDic[ant]['Dia'] + antDic[otherAnt]['Dia']) - projSep[other_index])/projSep[other_index] for other_index, otherAnt in enumerate(nearAntList)] 
        antDic[ant]['Shadow'] = np.max(np.array([shadowFrac[other_index] if shadowFrac[other_index] > 0 and frontA[other_index] > 0 else 0 for other_index, otherAnt in enumerate(nearAntList)]))
    return antDic
#-------- Apply Tsys calibration to visibilities
def applyTsysCal(prefix, BandName, BandbpSPW, scanDic, SSODic, XspecList):
    from interferometry import ANT0, ANT1, Ant2Bl, kb, Tcmb, GetAntName, GetAntD, GetTemp, indexList, smoothValue
    from atmCal import residTskyTransfer0
    #---- Check antenna list
    antList = GetAntName(prefix + '.ms')
    antNum = len(antList); blNum = int(antNum* (antNum - 1)/2)
    TrxAntList = np.load('%s-%s.TrxAnt.npy' % (prefix, BandName))
    antDia = GetAntD(TrxAntList)
    useAnt = indexList(TrxAntList,antList); useAntNum = len(useAnt); useBlNum = int(useAntNum* (useAntNum - 1)/2)
    ant0, ant1 = ANT0[0:useBlNum], ANT1[0:useBlNum]
    useBlMap = [Ant2Bl(useAnt[ant0[bl_index]], useAnt[ant1[bl_index]])  for bl_index in list(range(useBlNum))]
    nominalAe = 0.72
    polXindex, polYindex = (np.arange(4)//2).tolist(), (np.arange(4)%2).tolist()
    #---- Time variables (mjdSec, EL)
    atmTime, atmEL = np.load('%s-%s.atmTime.npy' % (prefix, BandName)), np.load('%s-%s.EL.npy' % (prefix, BandName))
    atmSecZ = 1.0 / np.sin(atmEL)
    tempAtm = GetTemp(prefix + '.ms')
    TrxList, TskyList, Tau0List, TauEList, SPList = [], [], [], [], []
    #-------- Zenith optical depth
    for spw_index, spw in enumerate(BandbpSPW['spw']):
        TrxList  = TrxList  + [np.load('%s-%s-SPW%d.Trx.npy'  % (prefix, BandName, spw))]   # TrxList[spw] [pol, ch, ant, scan]
        TskyList = TskyList + [np.load('%s-%s-SPW%d.Tsky.npy'  % (prefix, BandName, spw))]  # TskyList[spw] [ch, ant, scan]
        Tsky = np.median(TskyList[spw_index], axis=1)   # Sky should be commom for pol and ant
        #-------- Suppress Band-edge outliers
        chNum = Tsky.shape[0]
        Tsky[0:int(0.05*chNum)] = Tsky[int(0.05*chNum)]
        Tsky[-int(0.05*chNum):] = Tsky[int(-0.05*chNum)-1]
        initTau0 = -np.log(1 - np.median(Tsky)/tempAtm) / np.median(atmSecZ)
        if np.max(atmSecZ) - np.min(atmSecZ) > 0.1:
            Tau0List = Tau0List + [np.array([scipy.optimize.leastsq(residTskyTransfer0, [initTau0], args=(tempAtm, atmSecZ, Tsky[ch], atmEL))[0][0] for ch in range(Tsky.shape[0])])]
            Tau0Med = np.median(Tau0List[spw_index])
            TauEList = TauEList + [residTskyTransfer0([Tau0Med], tempAtm, atmSecZ, np.median(Tsky, axis=0), atmEL) / (tempAtm - Tcmb)* np.exp(-Tau0Med* atmSecZ) / atmSecZ]
            SPList = SPList + [tauSMTH(atmTime, TauEList[spw_index])]   # interpolation along time
        else:
            Tau0List = Tau0List + [-np.log(1.0 - np.median(Tsky, axis=1)/tempAtm)/np.median(atmSecZ)]
            TauEList = TauEList + [np.array(0.0)]
    #-------- Applying Tsys and attenuation correction for each scan
    for scan_index, scan in enumerate(scanDic.keys()):
        scanTau = []
        TsysScanDic = dict(zip(TrxAntList, [[]]* len(TrxAntList)))
        source = scanDic[scan]['source']
        for spw_index, spw in enumerate(BandbpSPW['spw']):
            chNum = BandbpSPW['chNum'][spw_index]
            TrxAnt = np.median(TrxList[spw_index], axis=3).transpose(1, 2, 0) 
            #-------- Suppress Band-edge outliers
            TrxAnt[0:int(0.05*chNum)] = TrxAnt[int(0.05*chNum)]
            TrxAnt[-int(0.05*chNum):] = TrxAnt[int(-0.05*chNum)-1]
            #-------- On-source excess (Tant) added to Tsys
            StokesI = SSODic[source][1][spw_index] if source in SSOCatalog else scanDic[scan]['I'] 
            Tant = StokesI* nominalAe* np.pi* antDia**2 / (8.0* kb)                     # Antenna temperature of SSO
            #-------- Interpolated optical depth in the scan
            Tau0SP = np.outer(Tau0List[spw_index], np.ones(len(scanDic[scan]['mjdSec'])))
            if np.max(atmSecZ) - np.min(atmSecZ) > 0.1: Tau0SP += scipy.interpolate.splev(scanDic[scan]['mjdSec'], SPList[spw_index])
            secZ = 1.0 / np.sin(scanDic[scan]['EL'])                           # Airmass
            scanTau = scanTau + [Tau0SP * secZ]             # Optical depth at the elevation
            exp_Tau = np.exp(-scanTau[spw_index])           # Atmospheric attenuation
            atmCorrect = np.mean(1.0 / exp_Tau, axis=1)     # Correction for atmospheric attenuation
            TsysScan = (Tcmb + Tant + (TrxAnt.transpose(2,1,0)* atmCorrect + tempAtm* (atmCorrect - 1.0)).transpose(0,2,1)).transpose(2,0,1)
            #-------- Tsys correction
            Xspec = XspecList[spw_index][scan_index][:,:,useBlMap].transpose(3,2,0,1)* np.sqrt(abs(TsysScan[ant0][:,polXindex]* TsysScan[ant1][:,polYindex]))
            XspecList[spw_index][scan_index][:,:,useBlMap] = Xspec.transpose(2,3,1,0)
            for ant_index, ant in enumerate(TrxAntList): TsysScanDic[ant] = TsysScanDic[ant] + [TsysScan[ant_index]]
        scanDic[scan]['Tau']  = scanTau
        scanDic[scan]['Tsys'] = TsysScanDic
    #
    return scanDic, XspecList
#
def aprioriSEFD(Ae, EL, TrxSpec, Tau0Spec):
    secZ = 1.0 / np.sin(EL)
    zenithTau = Tau0spec + scipy.interpolate.splev(np.median(timeStamp), exTauSP) + Tau0Coef[spw_index][0] + Tau0Coef[spw_index][1]*secZ
    exp_Tau = np.exp(-zenithTau * secZ )
    TsysEQScan = np.mean(TrxList[spw_index].transpose(2,0,1)[:,:,chRange] + Tcmb*exp_Tau[chRange] + tempAtm* (1.0 - exp_Tau[chRange]), axis=2)[Trx2antMap] # [antMap, pol]
    return 2.0* kb* TsysEQScan.T / Ae
#
#-------- Disk Visibility
def diskVis(diskRadius, u):
    # diskRadius : radius of planet disk [rad]
    # u          : spatial frequency (= baseline / wavelength)
    argument = 2.0* np.pi* u* diskRadius
    return 2.0* scipy.special.jn(1, argument) / argument
#
#-------- Disk Visibility with primary beam correction, u must be smaller than 0.3/diskRadius
def diskVisBeam(diskShape, u, v, primaryBeam):
    # diskShape  : planet disk diameter [MajorAxis, MinorAxis, PA] (rad)
    # u,v        : spatial frequency (= baseline / wavelength)
    # primaryBeam: FWHM of primary beam [rad]
    from interferometry import beamF
    cs, sn = np.cos(diskShape[2]), np.sin(diskShape[2])
    diskRadius = 0.5* np.sqrt(diskShape[0]* diskShape[1])
    DSmaj = 1.0 / np.sqrt( (0.30585 / diskShape[0])**2 + 2.0* np.log(2.0)/(np.pi* primaryBeam)**2 )    # Primary-beam correction
    DSmin = 1.0 / np.sqrt( (0.30585 / diskShape[1])**2 + 2.0* np.log(2.0)/(np.pi* primaryBeam)**2 )    # Primary-beam correction
    uvDisp = (DSmin*(u* cs - v* sn))**2 + (DSmaj*(u* sn + v* cs))**2
    return beamF(diskRadius/primaryBeam)* np.exp(-0.5* uvDisp)
#
#-------- Apertue effciency measurements using Solar System Objects
def SSOAe(antList, spwDic, uvw, scanDic, SSODic, XSList):
    # antList   : List of antenna name
    # spwDic    : SPW dictionary ['spw', 'freq', 'chNum', 'chRange', 'BW']
    # uvw       : baseline vector [m]
    # scanDic   : scan dictionary ['msfile', 'source', 'mjdSec', 'EL', 'PA', 'I', 'QCpUS', 'Tau', 'Tsys', 'Gain']
    # SSODic    : SSO dictionary
    # XSList    : Cross Correlation XspecList[spw][pol, ch, bl, time]
    from interferometry import GetAntD, Bl2Ant, ANT0, ANT1, kb, gainComplex
    SSOname = scanDic['source']
    text_sd = ' Flux Calibrator : %10s EL=%.1f' % (SSOname, 180.0*np.median(scanDic['EL'])/np.pi)
    uvDist = np.sqrt(uvw[0]**2 + uvw[1]**2)
    UVlimit = 0.25 / SSODic[SSOname][2][0]          # Maximum usable uv distance [lambda]
    text_sd = text_sd + ' uv limit = %5.0f klambda' % (UVlimit*1.0e-3); print(text_sd)
    antNum, blNum, antDia = len(antList), len(uvDist), GetAntD(antList)
    ant0, ant1 = ANT0[0:blNum], ANT1[0:blNum]
    #-------- Check usable antenna/baselines
    uvFlag = np.ones(blNum)
    for spw_index, spw in enumerate(spwDic['spw']):
        chRange = spwDic['chRange'][spw_index]
        centerFreq = np.mean(spwDic['freq'][spw_index][chRange])
        uvFlag[np.where( uvDist > UVlimit* 299792458 / centerFreq)[0].tolist()] *= 0.0
    #
    SAant = [0] + [Bl2Ant(bl_index)[0] for bl_index in np.where(uvFlag > 0.1)[0].tolist() if Bl2Ant(bl_index)[1] == 0]
    SAbl  = [bl_index for bl_index in list(range(blNum)) if Bl2Ant(bl_index)[0] in SAant and Bl2Ant(bl_index)[1] in SAant]
    if len(SAant) < 4: return #  Too few antennas
    #-------- Baseline map
    print('Subarray : ',); print(antList[SAant])
    for ant_index in list(range(1, antNum)):
        text_sd = antList[ant_index] + ' : '; print(text_sd, end='')
        blList = [bl_index for bl_index in list(range(blNum)) if Bl2Ant(bl_index)[0] == ant_index]
        for bl_index in blList:
            if uvFlag[bl_index] < 1.0: text_sd = '\033[91m%4.0f\033[0m' % (uvDist[bl_index])
            else: text_sd = '%4.0f' % (uvDist[bl_index])
            print(text_sd, end=' ')
        print('')
    print('       ', end='')
    for ant_index, ant in enumerate(antList[0:antNum-1]): print(ant, end=' ')
    print('')
    AeSPW, WgSPW, SSOmodel = [], [], []
    #-------- Aperture Efficiency
    for spw_index, spw in enumerate(spwDic['spw']):
        uvWave = uvw[0:2,:] * centerFreq / 299792458    # UV distance in wavelength
        primaryBeam = 1.13* 299792458 / (np.pi * antDia* centerFreq)
        SSOmodelVis = SSODic[SSOname][1][spw_index]*  diskVisBeam(SSODic[SSOname][2], uvWave[0], uvWave[1], primaryBeam[ant0]* primaryBeam[ant0]* np.sqrt(2.0 / (primaryBeam[ant0]**2 + primaryBeam[ant1]**2))) + 1.0e-9
        VisChav = np.mean(XSList[spw_index][:,chRange][:,:,SAbl], axis=(1,3)) / SSOmodelVis[SAbl]
        Gain = np.apply_along_axis(gainComplex, 1, VisChav)  # Gain[pol, ant]
        Aeff = 8.0* kb* abs(Gain)**2 / (np.pi* antDia[SAant]**2)
        Ae, Wg = np.zeros([antNum, 2]), np.zeros([antNum, 2])
        for ant_index, SA in enumerate(SAant):
            Ae[SA] = Aeff[:, ant_index]
            Wg[SA] = np.sign(Aeff[:, ant_index])* np.median(abs(SSOmodelVis))
        if np.median(Ae) > 0.99 : Wg *= 1.0e-6
        if np.median(Ae) < 0.19 : Wg *= 1.0e-2
        if np.median(scanDic['EL']) < ELshadow : Wg *= 0.01
        AeSPW = AeSPW + [Ae]
        WgSPW = WgSPW + [Wg]
        SSOmodel = SSOmodel + [SSOmodelVis]
    FscaleDic = {
        'Ae'    : AeSPW,
        'Wg'    : WgSPW,
        'model' : SSOmodel}
    return FscaleDic
#-------- Average Ae among multiple SSOs
def averageAe(FscaleDic, spwList):
    AeList, WgList = [], []
    for SSO_index, SSOname in enumerate(FscaleDic.keys()):
        if FscaleDic[SSOname] is None: continue
        if np.median(np.array(FscaleDic[SSOname]['Wg'])) < 1.0e-9: continue
        AeSPW = FscaleDic[SSOname]['Ae']; AeList = AeList + [np.array(AeSPW)]
        WgSPW = FscaleDic[SSOname]['Wg']; WgList = WgList + [np.array(WgSPW)]
    return  (np.sum(np.array(WgList) * np.array(AeList), axis=0)/(np.sum(np.array(WgList), axis=0)+1.0e-9)).transpose(1,2,0)   # Aeff[ant, pol, spw]
#-------- Transfer and equalize aperture efficiencies
def AeTransfer(VisChav, Aeff, antDia):
    from interferometry import Bl2Ant, gainComplexVec
    blNum = VisChav.shape[1]
    SAant = np.where(np.min(Aeff, axis=1) > 0.25)[0].tolist()
    SAbl  = [bl_index for bl_index in list(range(blNum)) if Bl2Ant(bl_index)[0] in SAant and Bl2Ant(bl_index)[1] in SAant]
    GainSA  = gainComplexVec(VisChav[:,SAbl].T)
    GainAll = gainComplexVec(VisChav.T)
    scaleFlux = np.median(abs(GainSA**2).T / (antDia[SAant]**2 * Aeff[SAant].T), axis=1)
    return ((abs(GainAll)**2 / scaleFlux).T / (antDia**2)).T
#
#-------- Gain scaling
def GainScale(Aeff, antDia, polVis):
    # Aeff :    Aperure Efficiency Table
    # antDia :  Antenna diameter
    # polVis[pol,bl,time] :    cross power spectra
    from interferometry import kb, ANT0, ANT1
    polXindex, polYindex = (np.arange(4)//2).tolist(), (np.arange(4)%2).tolist()
    blNum = polVis.shape[1]
    ant0, ant1 = ANT0[0:blNum], ANT1[0:blNum]
    fluxScale = np.sqrt( 2.0* kb / (0.25* np.pi* antDia**2* Aeff.T))
    return (polVis.transpose(2,0,1)* fluxScale[polYindex][:,ant0]* fluxScale[polXindex][:,ant1]).transpose(1,2,0)
#-------- Stokes parameters
def Vis2Stokes(VisChav, Dcat, PA):
    # VisChav   : channel-averaged visibiliities [pol, bl, time]
    # Dcat      : D-term [ant, pol]
    # PA        : Parallactic Angle with respect to X-feed [time]
    from interferometry import ANT0, ANT1, InvMullerMatrix, InvPAVector
    PAnum = len(PA)
    PS = InvPAVector(PA, np.ones(PAnum))
    blNum = VisChav.shape[1]
    Stokes = np.zeros([4,blNum], dtype=complex)
    for bl_index in list(range(blNum)):
        Minv = InvMullerMatrix(Dcat[ANT1[bl_index], 0], Dcat[ANT1[bl_index], 1], Dcat[ANT0[bl_index], 0], Dcat[ANT0[bl_index], 1])
        Stokes[:,bl_index] = PS.reshape(4, 4*PAnum).dot(Minv.dot( VisChav[:,bl_index]).reshape(4*PAnum)) / PAnum
    #
    return Stokes
#-------- Linear regression for visibility-baseline relation
def lmStokes(StokesVis, uvDist):
    StokesFlux, StokesErr, blNum = np.zeros([4]), np.ones([4]), len(uvDist)
    #-------- 1st look
    #tmpWeight = np.ones(blNum) / np.std(StokesVis[0].imag)
    tmpWeight = np.exp(-uvDist/max(uvDist)) / np.std(StokesVis[0].imag)
    tmpWeight[np.where(abs(StokesVis[0]) < np.median(abs(StokesVis[0])))[0].tolist()] *= 0.5
    tmpWeight[np.where(abs(StokesVis[0]) < np.percentile(abs(StokesVis[0]), 25))[0].tolist()] *= 0.5
    coef, cov = np.polyfit(uvDist, abs(StokesVis[0]), deg=1, w=tmpWeight, cov=True) # small weights for short baselines (shadowed?)
    residVis = abs(StokesVis[0]) - np.polyval(coef, uvDist)
    #-------- 2nd look
    tmpWeight = np.ones(blNum) / np.std(StokesVis[0].imag)
    #tmpWeight[np.where(abs(residVis) > 3.0*np.sqrt(cov[1,1]))[0].tolist()] *= 0.5
    #tmpWeight[np.where(abs(residVis) > 5.0*np.sqrt(cov[1,1]))[0].tolist()] *= 0.5
    tmpWeight[np.where(abs(residVis) > 10.0*np.sqrt(cov[1,1]))[0].tolist()] *= 0.1
    coef, cov = np.polyfit(uvDist, abs(StokesVis[0]), deg=1, w=tmpWeight, cov=True); coef_err = np.sqrt(np.diag(cov))
    if abs(coef[0]) < 3.0* coef_err[0] and coef[0]*np.max(uvDist) < -0.75*coef[1]: coef[0], coef[1] = 0.0, np.median(abs(StokesVis[0]))
    StokesFlux[0], StokesSlope, StokesErr[0] = coef[1], coef[0], coef_err[1]
    for pol_index in [1,2,3]:
        coef, cov = np.polyfit(uvDist, StokesFlux[0]*StokesVis[pol_index].real / abs(StokesVis[0].real), deg=0, w=tmpWeight, cov=True)
        StokesFlux[pol_index], StokesErr[pol_index] = coef[0], np.sqrt(cov[0,0])
    return StokesFlux, StokesSlope, StokesErr
#-------- Smooth time-variable Tau
def tauSMTH( timeSample, TauE ):
    if len(timeSample) > 5:
        SplineWeight = np.ones(len(timeSample) + 8)
        flagIndex = (np.where(abs(TauE - np.median(TauE))/np.std(TauE) > 3.0)[0] + 2).tolist()
        SplineWeight[flagIndex] = 0.1
        tempTime = np.append([timeSample[0]-100.0, timeSample[0]-20.0, timeSample[0]-10.0], np.append(timeSample, [timeSample[-1]+10.0, timeSample[-1]+20.0, timeSample[-1]+100.0, timeSample[-1]+200.0, timeSample[-1]+300.0]))
        tempTauE = np.append([TauE[0], TauE[0], TauE[0]], np.append(TauE, [TauE[-1], TauE[-1], TauE[-1], TauE[-1], TauE[-1]]))
        smthTau = scipy.interpolate.splrep(tempTime, tempTauE, k=3, w=SplineWeight, t=timeSample[1:-1])
    else:
        tempTime = np.arange(np.min(timeSample) - 3600.0,  np.max(timeSample) + 3600.0, 300.0)
        tempTauE = np.repeat(np.median(TauE), len(tempTime))
        smthTau = scipy.interpolate.splrep(tempTime, tempTauE, k=3)
    return smthTau
#
