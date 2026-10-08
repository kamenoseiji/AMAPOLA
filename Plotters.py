import os
import math
from casatools import quanta as qatool
import numpy as np
import scipy
import datetime
from interferometry import indexList, GetChNum, bunchVec, delay_search, Bl2Ant, Ant2Bl, RADDEG, Tcmb, mjd2utc, speed_of_light
from Grid import tauSMTH, SSOCatalog
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
qa = qatool()
RADSEC = 180*3600/np.pi         # 206264.80624709636 arcsec/rad
polColor = ['b', 'g', 'm', 'y']
polName = ['X', 'Y', 'XY', 'YX']
StokesComponents = ['I', 'Q', 'U', 'V']
#-------- Set Color Map
lineCmap = plt.get_cmap('Set1')
#-------- Plot baseline-AllanVariance map
def plotBLAV(prefix, antList, spw, AV_bl):
    pp = PdfPages('AV_%s-SPW%d.pdf' % (prefix, spw))
    figAV = plt.figure(figsize = (8, 8))
    figAV.suptitle('Allan Variance %s SPW%d' % (prefix, spw))
    axAV = figAV.add_subplot(1, 1, 1)
    antNum, blNum = len(antList), AV_bl.shape[1]
    AVmap = np.zeros([2, antNum, antNum])
    for bl_index in list(range(blNum)):
        ants = Bl2Ant(bl_index)
        AVmap[0, ants[1], ants[0]] = AV_bl[0, bl_index]  # pol-X
        AVmap[1, ants[0], ants[1]] = AV_bl[1, bl_index]  # pol-Y
    #
    AVmap = np.sum(AVmap, axis=0)
    for ant_index, ant in enumerate(antList): AVmap[ant_index, ant_index] = 0.0
    imAV = axAV.imshow(AVmap, cmap='coolwarm', vmin=0.0, vmax=0.05)
    figAV.colorbar(imAV, ax=axAV)
    axAV.plot([0,antNum-1], [0,antNum-1], color='w')
    axAV.set_xticks(list(range(antNum))); axAV.set_yticks( list(range(antNum)))
    axAV.set_xticklabels(antList.tolist()); axAV.set_yticklabels(antList.tolist())
    figAV.savefig(pp, format='pdf')
    plt.close('all')
    pp.close()
    return
#-------- Plot Cross power spectrum
def plotSP(pp, prefix, antList, spwList, freqList, BPList, plotMin=0.0, plotMax=1.5, delayMessage=False):
    antNum, spwNum = len(antList), len(spwList)
    delayAnt = np.zeros([antNum, spwNum, 2])
    figAnt = plt.figure(figsize = (11, 8))
    figAnt.suptitle(prefix + ' Bandpass')
    figAnt.text(0.45, 0.05, 'Frequency [GHz]')
    figAnt.text(0.03, 0.45, 'Bandpass Amplitude and Phase', rotation=90)
    #-------- Plot BP
    text_delay = 'Ant  '
    for spw_index, spw in enumerate(spwList):
        text_delay = text_delay + ' |SPW%02d'%spw
        text_delay = text_delay + '  ' + polName[0]
        if BPList[spw_index].shape[1] == 2: text_delay = text_delay + '         ' + polName[1]
    if delayMessage: print(text_delay)
    text_delay = '------+'
    for spw_index, spw in enumerate(spwList): text_delay = text_delay + '--- delay [ns] ----+'
    if delayMessage: print(text_delay)
    for ant_index, antName in enumerate(antList):
        if ant_index > 0:
            for PL in AmpList: figAnt.delaxes(PL)
            for PL in PhsList: figAnt.delaxes(PL)
        #
        AmpList, PhsList = [], []
        text_delay = antName + '  |'
        for spw_index, spw in enumerate(spwList):
            Freq = freqList[spw_index]
            plotFreq = 1.0e-9 * Freq
            chNum = len(Freq); chRange = list(range(int(0.10*chNum), int(0.95*chNum)))
            BW = Freq[chRange[-1]] - Freq[chRange[0]]
            AmpPL = figAnt.add_subplot(2, spwNum, spw_index + 1 )
            PhsPL = figAnt.add_subplot(2, spwNum, spwNum + spw_index + 1 )
            AmpList = AmpList + [AmpPL]
            PhsList = PhsList + [PhsPL]
            ppolNum = BPList[spw_index].shape[1]
            for pol_index in list(range(ppolNum)):
                plotBandpass = BPList[spw_index][ant_index,pol_index]
                delay_ant, delaySNR = delay_search(plotBandpass[chRange])
                delayAnt[ant_index, spw_index, pol_index] = 0.5e9* delay_ant / BW   # delay in [ns]
                AmpPL.fill_between(plotFreq[chRange], plotMin, plotMax, color='yellow', alpha=0.1)
                AmpPL.step(plotFreq, abs(plotBandpass), color=polColor[pol_index], where='mid', label = 'Pol-%s' % (polName[pol_index]))
                text_delay = text_delay + '  %+.4f ' % (0.5e9* delay_ant/ BW)
                PhsPL.fill_between(plotFreq[chRange], -np.pi, np.pi, color='yellow', alpha=0.1)
                PhsPL.plot(plotFreq, np.angle(plotBandpass), '.', color=polColor[pol_index], label = 'Pol-%s %+.3f ns' % (polName[pol_index], -0.5e9* delay_ant / BW))
            #
            if spw_index == 0: AmpPL.set_title(antList[ant_index])
            AmpPL.axis([np.min(plotFreq), np.max(plotFreq), plotMin, plotMax])
            AmpPL.tick_params(axis='both', labelsize=6)
            AmpPL.legend(loc = 'lower left', prop={'size' :7}, numpoints = 1)
            AmpPL.text( np.min(plotFreq), 0.9* plotMax + 0.1* plotMin, 'SPW=%d Amp' % (spwList[spw_index]))
            PhsPL.axis([np.min(plotFreq), np.max(plotFreq), -np.pi, np.pi])
            PhsPL.tick_params(axis='both', labelsize=6)
            PhsPL.legend(loc = 'lower left', prop={'size' :7}, numpoints = 1)
            PhsPL.text( np.min(plotFreq), 2.5, 'SPW=%s Phs' % (str(spw)))
        #
        if delayMessage: print(text_delay)
        figAnt.savefig(pp, format='pdf')
    plt.close('all')
    pp.close()
    return delayAnt
#-------- Plot Tsys and Trx Spectrum
def plotTsysDic(prefix, TsysDic):
    bandName = TsysDic[next(iter(TsysDic))]['Band']
    pp = PdfPages('TSYS_%s.%s.pdf' % (prefix, bandName))
    #-------- Plots for Tsys spectra
    scanList = list(TsysDic.keys()); scanNum = len(scanList)
    antList  = TsysDic[scanList[0]]['antList']; antNum = len(antList)
    #-------- Prepare Plots
    figAnt = plt.figure(figsize = (8, 11))
    figAnt.suptitle(prefix)
    figAnt.text(0.45, 0.05, 'Frequency [GHz]')
    figAnt.text(0.03, 0.42, 'Tsys (X:blue, Y:green) and Trec (X:magenta, Y:yellow) [K]', rotation=90)
    #-------- Page for each antenna
    for ant_index, ant in enumerate(antList):
        if ant_index > 0:
            for PL in TsysPL: figAnt.delaxes(PL)
        TsysPL, TsysMax = [], []
        #-------- To determine scaling max
        for scan_index, scan in enumerate(scanList):
            spwNum = len(TsysDic[scan]['spwList'])
            for spw_index, spw in enumerate(TsysDic[scan]['spwList']):
                index = spwNum* ant_index + spw_index
                TsysMax = TsysMax + [np.percentile(TsysDic[scan]['Tsys'][:,:,index], 75)]
        plotMax = 1.5* np.max(np.array(TsysMax))
        for scan_index, scan in enumerate(scanList):
            spwNum = len(TsysDic[scan]['spwList'])
            for spw_index, spw in enumerate(TsysDic[scan]['spwList']):
                index = spwNum* ant_index + spw_index
                TRX, TSYS, FREQ = TsysDic[scan]['Trx'][:,:,index], TsysDic[scan]['Tsys'][:,:,index], TsysDic[scan]['freq'][spw_index]
                polNum = TRX.shape[0]
                currentPL = figAnt.add_subplot(scanNum, spwNum, spwNum* scan_index + spw_index + 1 )
                TsysPL = TsysPL + [currentPL]
                timeLabel = mjd2utc(TsysDic[scan]['mjdSec'])[:-4]
                for pol_index in range(polNum):
                    currentPL.step(FREQ, TSYS[pol_index], where='mid', color=polColor[pol_index], label = 'Tsys Pol '+ polName[pol_index])
                    currentPL.step(FREQ, TRX[pol_index],  color=polColor[pol_index+2], where='mid', label = 'Trec Pol ' + polName[pol_index])
                currentPL.axis([np.min(FREQ), np.max(FREQ), 0.0, plotMax])
                currentPL.tick_params(axis='both', labelsize=3, direction='in')
                currentPL.text(0.25*np.min(FREQ) + 0.75*np.max(FREQ), 0.8* plotMax, 'SPW %d' % (spw), fontsize='6')
                if scan_index == 0 and spw_index == 0: currentPL.text(1.2* np.min(FREQ) - 0.2* np.max(FREQ), (0.9 + 0.075*scanNum)*plotMax, ant, fontsize='16')
                if spw_index == 0: currentPL.text(np.min(FREQ), 0.8* plotMax, timeLabel, fontsize='6')
                else: currentPL.set_yticklabels([])
        figAnt.savefig(pp, format='pdf')
    for PL in TsysPL: figAnt.delaxes(PL)
    plt.close('all')
    pp.close()
    del(TsysPL)
    del(figAnt)
    return
#-------- Plot XY-phase spectra
def plotXYP(pp, prefix, spwList, XYspec, XYdelay, bunchNum=1):
    spwNum = len(spwList)
    figXYP = plt.figure(figsize = (11, 8))
    figXYP.suptitle(prefix + ' XY Phase')
    figXYP.text(0.45, 0.05, 'Frequency [GHz]')
    figXYP.text(0.03, 0.45, 'XY Phase [deg]', rotation=90)
    for spw_index, spw in enumerate(spwList):
        chNum, chWid, Freq = GetChNum(prefix + '.ms', spw); Freq = 1.0e-9* bunchVec(Freq, bunchNum)  # GHz
        PhsPL = figXYP.add_subplot(1, spwNum, spw_index + 1)
        XYP  = XYspec[spw_index]
        PhsPL.plot( Freq, np.angle(XYP)*RADDEG, '.', label = 'SPW%2d: XYdelay=%+.3f ns' % (spw, -XYdelay[spw_index]))
        PhsPL.axis([np.min(Freq), np.max(Freq), -180.0, 180.0])
        PhsPL.tick_params(axis='both', labelsize=6)
        PhsPL.legend(loc = 'lower left', prop={'size' :7}, numpoints = 1)
    #
    #plt.show()
    figXYP.savefig(pp, format='pdf')
    plt.close('all')
    pp.close()
    return
#-------- Plot Stokes parameters vs uv distance
def plotFL(pp, scanDic, SPWDic):
    Pcolor   = ['black', 'blue', 'red', 'green']
    for scan_index, scan in enumerate(scanDic.keys()):
        scanFlag = scanDic[scan]['scanFlag']
        if len(scanFlag) == 0: continue
        figFL, axes = plt.subplots(3, 4, figsize = (11, 8))
        figFL.suptitle(scanDic[scan]['msfile'][:-3])
        figFL.text(0.45, 0.05, 'Projected baseline [m]')
        UVW = scanDic[scan]['UVW'][:,:,scanFlag]; uvw = np.mean(UVW, axis=2); uvDist = np.sqrt(uvw[0]**2 + uvw[1]**2)
        sourceName = scanDic[scan]['source']
        text_src  = 'Scan%02d %010s EL=%4.1f deg' % (scan, sourceName, 180.0* np.median(scanDic[scan]['EL'][scanFlag])/np.pi)
        timeLabel = qa.time('%fs' % np.median(scanDic[scan]['mjdSec'][scanFlag]), form='ymd')[0] + ' SA=%.1f' % (scanDic[scan]['SA']) + ' deg.'
        #---- Display results
        uvMin, uvMax, IMax = min(uvDist), max(uvDist), np.max(scanDic[scan]['ScanFlux'][:,0])
        axes[0,0].text(0.0, 1.15*180, text_src)
        axes[0,3].text(0.0, 1.15*180, timeLabel)
        blFlag = scanDic[scan]['blFlag']
        for spw_index, spw in enumerate(SPWDic['spw']):
            StokesVis = scanDic[scan]['StokesVis'][spw_index]
            ScanFlux = scanDic[scan]['ScanFlux'][spw_index]
            ScanSlope = scanDic[scan]['uvSlope'][spw_index]* ScanFlux / ScanFlux[0]
            #-------- Plot Stokes visibilities
            axes[0,spw_index].plot( np.array([0.0, uvMax]), np.array([0.0, 0.0]), '-', color='grey')        # phase-0 line
            axes[1,spw_index].plot( np.array([0.0, uvMax]), np.array([ScanFlux[0], ScanFlux[0] + uvMax* ScanSlope[0]]), '-', color=Pcolor[0])    # Stokes I fitline
            axes[2,spw_index].plot( np.array([0.0, uvMax]), np.array([ScanFlux[1], ScanFlux[1] + uvMax* ScanSlope[1]]), '-', color=Pcolor[1])    # Stokes Q fitline
            axes[2,spw_index].plot( np.array([0.0, uvMax]), np.array([ScanFlux[2], ScanFlux[2] + uvMax* ScanSlope[2]]), '-', color=Pcolor[2])    # Stokes U fitline
            axes[2,spw_index].plot( np.array([0.0, uvMax]), np.array([ScanFlux[3], ScanFlux[3] + uvMax* ScanSlope[3]]), '-', color=Pcolor[3])    # Stokes V fitline
            #
            axes[0,spw_index].plot( uvDist[blFlag], 180.0* np.angle(StokesVis[0][blFlag])/ np.pi, '.', label=StokesComponents[0], color=Pcolor[0])  # Plot visibility phase
            axes[1,spw_index].plot( uvDist[blFlag], StokesVis[0][blFlag].real, '.', label=StokesComponents[0], color=Pcolor[0])     # Plot Stokes I
            axes[2,spw_index].plot( uvDist[blFlag], StokesVis[1][blFlag].real, '.', label=StokesComponents[1], color=Pcolor[1])     # Plot Stokes Q
            axes[2,spw_index].plot( uvDist[blFlag], StokesVis[2][blFlag].real, '.', label=StokesComponents[2], color=Pcolor[2])     # Plot Stokes U
            axes[2,spw_index].plot( uvDist[blFlag], StokesVis[3][blFlag].real, '.', label=StokesComponents[3], color=Pcolor[3])     # Plot Stoees V
            #
            axes[0,spw_index].axis([0.0, uvMax, -180, 180])
            axes[1,spw_index].axis([0.0, uvMax, 0.0, 1.25*IMax])
            axes[2,spw_index].axis([0.0, uvMax, -0.25*IMax, 0.25*IMax])
            axes[0,spw_index].text(0.0, 1.02*180, 'SPW%2d %5.1f GHz' % (spw, 1.0e-9* np.median(SPWDic['freq'][spw_index])))
        #
        axes[0,0].set_ylabel('Phase [deg]')
        axes[1,0].set_ylabel('Stokes I [Jy]')
        axes[2,0].set_ylabel('Stokes Q,U,V [Jy]')
        axes[1,spw_index].legend(loc = 'best', prop={'size' :7}, numpoints = 1)
        axes[2,spw_index].legend(loc = 'best', prop={'size' :7}, numpoints = 1)
        figFL.savefig(pp, format='pdf')
    plt.close('all')
    pp.close()
    return
#-------- Plot Stokes parameters vs uv distance
def plotResidualMap(pp, scanDic, SPWDic):
    NyquistRatio = 4    # oversampling in (u,v) space
    repFreq = np.mean([np.median(SPWDic['freq'][spw_index]) for spw_index, spw in enumerate( SPWDic['spw'] )])
    for scan in scanDic.keys():
        figIM, axes = plt.subplots(2, 2, figsize = (11, 8), sharex=True, sharey=True, gridspec_kw={'right':0.9})
        figIM.suptitle(scanDic[scan]['source'])
        figIM.text(0.45, 0.05, 'Relative RA [arcsec]')
        figIM.text(0.03, 0.45, 'Relative DEC [arcsec]', rotation=90)
        timeLabel = qa.time('%fs' % np.median(scanDic[scan]['mjdSec']), form='ymd')[0]
        text_src  = '%s Scan%02d %s EL=%4.1f deg' % (scanDic[scan]['msfile'][:-3], scan, timeLabel, 180.0* np.median(scanDic[scan]['EL'])/np.pi)
        uv = repFreq* np.mean(scanDic[scan]['UVW'][:2], axis=2) / speed_of_light
        cellSize = RADSEC / np.sqrt(np.max(uv[0]**2 + uv[1]**2)) / (2* NyquistRatio)
        mapSize  = RADSEC / np.sqrt(np.min(uv[0]**2 + uv[1]**2))
        cellNum, blNum, spwNum = int(mapSize/cellSize), uv.shape[1], len(SPWDic['spw'])
        cellNum = 1 << (cellNum.bit_length() - 1)               # Round down to powers of 2
        cellNum = min(cellNum, 256)                             # Cap by 256 pix
        mapSize = cellNum* cellSize
        StokesMap = np.zeros([4, 2*cellNum, 2*cellNum])
        Map_l, Map_m = np.meshgrid(cellSize* np.array(range(-cellNum, cellNum)), cellSize* np.array(range(-cellNum, cellNum)))
        uvCellSize = 0.5*RADSEC/(cellSize*cellNum)
        for Stokes_index, Stokes in enumerate(StokesComponents): 
            UVMap, UVnum, UVcount = np.zeros([2*cellNum, 2*cellNum], dtype=complex), np.zeros([2*cellNum, 2*cellNum]), 0
            for spw_index, spw in enumerate(SPWDic['spw']):
                freq = np.median(SPWDic['freq'][spw_index])     # Hz
                uv = freq* np.mean(scanDic[scan]['UVW'][:2], axis=2) / speed_of_light
                StokesVis = (scanDic[scan]['StokesVis'][spw_index].T - scanDic[scan]['ScanFlux'][spw_index]).T
                #StokesVis = scanDic[scan]['StokesVis'][spw_index]
                uv_scale = uv/uvCellSize
                for bl_index in scanDic[scan]['blFlag']:
                    UVMap[(cellNum + round(uv_scale[1,bl_index]))^cellNum, (cellNum + round(uv_scale[0,bl_index]))^cellNum] += StokesVis[Stokes_index,bl_index]
                    UVMap[(cellNum - round(uv_scale[1,bl_index]))^cellNum, (cellNum - round(uv_scale[0,bl_index]))^cellNum] += StokesVis[Stokes_index,bl_index].conjugate()
                    UVnum[(cellNum + round(uv_scale[1,bl_index]))^cellNum, (cellNum + round(uv_scale[0,bl_index]))^cellNum] += 1
                    UVnum[(cellNum - round(uv_scale[1,bl_index]))^cellNum, (cellNum - round(uv_scale[0,bl_index]))^cellNum] += 1
            UVMap[np.where( UVnum > 0 )] /= UVnum[np.where( UVnum > 0 )]
            StokesMap[Stokes_index] = np.fft.fftshift(np.fft.fft2(UVMap)).real
        StokesMap *= (1.0/len(UVMap[np.where( UVnum > 0 )]))     # Scaling
        #-------- Plot residual Stokes map
        StokesRMS = np.std( StokesMap[3] )      # image rms in StokesV
        for index_Stokes, ax in enumerate(axes.ravel()):
            peakPos = np.array((np.where(StokesMap[0] == np.max(StokesMap[0]))))[:,0]   # position of peak in Stokes I
            peakI   = StokesMap[0][peakPos[0],peakPos[1]]
            SubComponent = (peakI > 7*np.std(np.std(StokesMap[0]))) & ((peakPos - np.array([cellNum,cellNum])).dot(peakPos - np.array([cellNum,cellNum])) > NyquistRatio**2)  # Secondary peak
            if SubComponent: text_src = text_src + ' Secondary at (%.1f, %.1f) arcsec' % (Map_l[peakPos[0],peakPos[1]], Map_m[peakPos[0],peakPos[1]])
            if index_Stokes == 0:
                im = ax.contourf(Map_l, Map_m, StokesMap[0], levels=10, vmin=-StokesRMS, vmax=np.max(StokesMap[0]), cmap='viridis')
                colorvar_I = figIM.add_axes([0.45, 0.50, 0.02, 0.40])
                figIM.colorbar(im, cax=colorvar_I)
                if SubComponent: ax.text(Map_l[peakPos[0],peakPos[1]], Map_m[peakPos[0],peakPos[1]], ' %.1f mJy' % (1000* StokesMap[0,peakPos[0],peakPos[1]]))
                ax.text(mapSize, 1.25*mapSize, text_src)
            else:
                im = ax.contourf(Map_l, Map_m, StokesMap[index_Stokes], levels=10, vmin=-5*StokesRMS, vmax=5*StokesRMS, cmap='viridis')
                if SubComponent: ax.text(Map_l[peakPos[0],peakPos[1]], Map_m[peakPos[0],peakPos[1]], ' %.1f mJy' % (1000* StokesMap[index_Stokes,peakPos[0],peakPos[1]]))
            ax.set_title('%s subtracted %.3f Jy' % (StokesComponents[index_Stokes], np.mean(scanDic[scan]['ScanFlux'][:,index_Stokes])))
            ax.plot(0.0, 0.0, 'w+')
            if SubComponent: ax.plot(Map_l[peakPos[0],peakPos[1]], Map_m[peakPos[0],peakPos[1]], 'r+')
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim( cellSize*cellNum, -cellSize*cellNum)
            ax.set_ylim(-cellSize*cellNum,  cellSize*cellNum)
        colorvar_P = figIM.add_axes([0.88, 0.10, 0.02, 0.40])
        figIM.colorbar(im, cax=colorvar_P)
        colorvar_P.set_title('Jy')
        figIM.savefig(pp, format='pdf')
    plt.close('all')
    pp.close()
    return
#-------- Plot XY cross correlation versus QU
def plotQUXY(pp, scanDic):
    figQU = plt.figure(figsize = (8, 11))
    figQU.text(0.25, 0.05, 'Linear polarization angle w.r.t. X-Feed [deg]')
    sourceList = list(set([scanDic[scan]['source'] for scan in scanDic.keys()]))
    ParaPolPL  = figQU.add_subplot( 2, 1, 1 )
    CrosPolPL  = figQU.add_subplot( 2, 1, 2 )
    cmap = plt.get_cmap("tab10")
    plotMax = 0.0
    for source_index, sourceName in enumerate(sourceList):
        if sourceName in SSOCatalog: continue
        scanList = [scan for scan in scanDic.keys() if 'scanVis' in scanDic[scan].keys() and scanDic[scan]['source'] == sourceName]
        if len(scanList) < 1: continue
        ThetaPlot, VisXX, VisYY, VisXY = [], [], [], []
        #-------- PA range to draw model
        for scan in scanList:
            Stokes = np.mean(scanDic[scan]['scanVis'], axis=1)
            EVPA = 0.5* np.arctan2(Stokes[2], Stokes[1])
            ThetaPlot = ThetaPlot + np.arctan(np.tan(scanDic[scan]['PA'] - EVPA)).tolist()
        ThetaPlot = np.array(ThetaPlot)
        ThetaMin, ThetaMax = np.min(ThetaPlot), np.max(ThetaPlot)
        ThetaRange = np.arange(ThetaMin, ThetaMax, 0.01)
        PArange = ThetaRange + EVPA
        #-------- Draw model
        CSrange, SNrange = np.cos(2.0*PArange), np.sin(2.0*PArange)
        UCmQS, QCpUS = Stokes[2]*CSrange - Stokes[1]* SNrange, Stokes[1]*CSrange + Stokes[2]* SNrange
        ParaPolPL.plot( RADDEG* ThetaRange, QCpUS, color=cmap(source_index), linestyle='dashed')
        ParaPolPL.plot( RADDEG* ThetaRange,-QCpUS, color=cmap(source_index), linestyle='dashdot')
        CrosPolPL.plot( RADDEG* ThetaRange, UCmQS, color=cmap(source_index), linestyle='solid')
        CrosPolPL.plot( RADDEG* ThetaRange, np.zeros(len(ThetaRange)), color=cmap(source_index), linestyle='dotted')
        #-------- Plot data points
        for scan in scanList:
            visChav = scanDic[scan]['visChav']   # visChav[Stokes, time]
            VisXX = VisXX + (abs(scanDic[scan]['visChav'][0]) - abs(np.mean(scanDic[scan]['visChav'][[0,3]], axis=0))).tolist()
            VisYY = VisYY + (abs(scanDic[scan]['visChav'][3]) - abs(np.mean(scanDic[scan]['visChav'][[0,3]], axis=0))).tolist()
            VisXY = VisXY + np.mean(scanDic[scan]['visChav'][[1,2]], axis=0).tolist()
            text_sd = 'Scan %d : %s' % (scan, qa.time('%fs' % (scanDic[scan]['mjdSec'][0]), form='fits', prec=6)[0][11:21])
            text_PA = RADDEG* np.arctan(np.tan((scanDic[scan]['PA'][0] - EVPA)))
            rotText, posText = 90, 'bottom'
            if text_PA < 0: rotText, posText = -90, 'top'
            CrosPolPL.text(text_PA, 0, text_sd, verticalalignment=posText, fontsize=6, rotation=rotText)
        VisXX, VisYY, VisXY, QCpUS, UCmQS = np.array(VisXX), np.array(VisYY), np.array(VisXY), np.array(QCpUS), np.array(UCmQS)
        ParaPolPL.plot( RADDEG* ThetaPlot, VisXX, 'x', fillstyle='full', color=cmap(source_index), label=sourceName + ' XX')
        ParaPolPL.plot( RADDEG* ThetaPlot, VisYY, '1', fillstyle='none', color=cmap(source_index), label=sourceName + ' YY')
        CrosPolPL.plot( RADDEG* ThetaPlot, VisXY.real, 'o', fillstyle='none', mew=0.2, color=cmap(source_index), label=sourceName + ' re')
        CrosPolPL.plot( RADDEG* ThetaPlot, VisXY.imag, ',', color=cmap(source_index), label=sourceName + ' im')
        plotMax = np.max([plotMax, np.max(abs(VisXX))])
        plotMax = np.max([plotMax, np.max(abs(VisYY))])
        plotMax = np.max([plotMax, np.max(abs(VisXY))])
    #
    ParaPolPL.axis([-90.0, 90.0, -1.2*plotMax, 1.2*plotMax])
    ParaPolPL.grid(linewidth=0.5, linestyle='dotted')
    ParaPolPL.set_ylabel('Parallel-hand correlation - Stokes I [Jy]')
    CrosPolPL.axis([-90.0, 90.0, -1.2*plotMax, 1.2*plotMax])
    CrosPolPL.grid(linewidth=0.5, linestyle='dotted')
    CrosPolPL.set_ylabel('Cross-hand correlation [Jy]')
    box = ParaPolPL.get_position()
    ParaPolPL.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ParaPolPL.legend(loc = 'upper right', prop={'size' :4.5}, ncol=2, numpoints=1, labelspacing=0.05, bbox_to_anchor = (1.3,1))
    box = CrosPolPL.get_position()
    CrosPolPL.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    CrosPolPL.legend(loc = 'upper right', prop={'size' :4.5}, ncol=2, numpoints=1, labelspacing=0.05, bbox_to_anchor = (1.3,1))
    figQU.suptitle('Cross-Pol. Correlations %s' % (scanDic[scan]['msfile']))
    figQU.savefig(pp, format='pdf')
    plt.close('all')
    pp.close()
    return
#
#-------- Tool 
def get_progressbar_str(progress):
    MAX_LEN = 48
    BAR_LEN = int(MAX_LEN * progress)
    return ('[' + '=' * BAR_LEN + ('>' if BAR_LEN < MAX_LEN else '') + ' ' * (MAX_LEN - BAR_LEN) + '] %.1f%%' % (progress * 100.))
#
