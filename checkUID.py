import os
import glob
import numpy as np
from datetime import datetime
from interferometry import GetAntName, GetBandNames
def replaceList( UID, status ):
    UID_text = UID.replace("___", r"\:\/\/").replace("_",r"\/")
    return('sed -e "s/new %s/%s %s/g" UID/UIDList > UID/UIDnew' % (UID_text, status, UID_text))
#
def execGridReduction(prefix):
    text_sd = 'casa -c ' + SCR_DIR + 'checkGridSurvey.py -u %s' % (prefix)
    print(text_sd)
    os.system(text_sd)
    return
#
def storeLog(prefix, bandName):
    print('scp ' + prefix + '-' + bandName + '-Flux.log ' + URI_Ingest)
    os.system('scp ' + prefix + '-' + bandName + '-Flux.log ' + URI_Ingest)
    os.system('rm -rf casa*.log')
    os.system('mv *.npy NPY/')
    os.system('mv *.pdf PDF/')
    os.system('mv *.log LOG/')
    os.system('rm -rf *.cl')
    return
#
SCR_DIR = os.environ['HOME'] + '/AMAPOLA/'                                      # Script path
URI_Ingest = 'skameno@ssh.alma.cl:/home/skameno/public_html/Grid/Stokes/'       # to store log files
UIDfile = open('./UID/UIDList', 'r')
UIDentry = UIDfile.readlines()
UIDfile.close()
UIDentry = list(set(UIDentry))
newEntry = [entry for entry in UIDentry if (entry.split()[0] == 'new') and ('B9' not in entry.split()[-1])]
FSUIDs = list(set([entry.split('/')[3] for entry in newEntry]))     # Each FSR
FSUIDs = sorted(FSUIDs, key=lambda x:int(x[1:], 16), reverse=True)                # new to old order
for FS in FSUIDs:
    newPrefix = 'uid___A002_' + FS
    FSentry = [entry for entry in newEntry if FS in entry]
    ArrayIDs = list(set([entry.split()[2] for entry in FSentry]))   # Each Array
    for Array in ArrayIDs:
        ARentry = [entry for entry in FSentry if Array in entry]
        EBList = [entry.split()[1].replace("/", "_").replace(":","_").replace(" ","") for entry in ARentry]
        #-------- Generage MS
        for prefix in EBList:                                       # Each EB
            if not os.path.isdir(prefix + '.ms'):
                if not os.path.isdir(prefix):
                    if not os.path.isdir('UID/' + prefix):
                        os.system('asdmExportLight %s' % (prefix))
                        if not os.path.isdir(prefix):
                            EBList.remove(prefix)
                            continue
                        os.system('mv %s UID/' % (prefix))
                    os.system('ln -s UID/' + prefix + ' .')
                importasdm(prefix)
        #-------- Check Number of Antennas in the array
        antList = GetAntName(EBList[0] + '.ms')
        if len(antList) < 4: 
            for prefix in EBList:
                text_sd = replaceList(prefix, 'fail')
                os.system(text_sd)
                os.system('mv ./UID/UIDnew ./UID/UIDList')
            continue
        #-------- ExecBock in the session
        EBdic = dict(zip(EBList, [[]]*len(EBList)))
        for index, prefix in enumerate(EBList):
            EBdic[prefix] = {
                'Band' : np.unique(GetBandNames(prefix + '.ms')).tolist(),
                'DT'   : datetime.strptime(ARentry[index].split()[3], "%Y-%m-%dT%H:%M:%S").timestamp() }
            #-------- Split multi-band EB into single-band EBs
            if len(EBdic[prefix]['Band']) > 1:
                text_sd = 'casa -c ' + SCR_DIR + 'splitEB.py -u %s' % (prefix)
                print(text_sd)
                os.system(text_sd)
                for bandName in EBdic[prefix]['Band']:
                    EBdic[prefix + '.split.' + bandName] = {
                        'Band' : bandName,
                        'DT'   : EBdic[prefix]['DT']}
                del EBdic[prefix]
        #-------- Classification by band
        bandList = np.unique(np.array([EBdic[prefix]['Band'] for prefix in EBdic.keys()])).tolist()
        for bandName in bandList:
            bandEBList = [EB for EB in EBdic.keys() if bandName in EBdic[EB]['Band']]
            #-------- Single EB (no concatenate)
            if len(bandEBList) == 1:
                execGridReduction(bandEBList[0])
                storeLog(bandEBList[0], bandName)
                text_sd = replaceList(bandEBList[0], 'done')
                os.system(text_sd)
                os.system('mv ./UID/UIDnew ./UID/UIDList')
                os.system('casa -c ' + SCR_DIR + 'AMAPOLA2SC.py -u %s' % (bandEBList[0]))
            #-------- Single EB (no concatenate)
            else:
                for prefix in bandEBList:
                    if 'split' not in prefix:
                        text_sd = 'casa -c ' + SCR_DIR + 'splitEB.py -u %s' % (prefix)
                        print(text_sd)
                        os.system(text_sd)
                        EBdic[prefix + '.split.' + bandName] = EBdic.pop(prefix)
                EBdic = dict(sorted(EBdic.items(), key=lambda item: item[1]['DT']))     # Sort by DT
                splitBandEBList = [EB for EB in EBdic.keys() if bandName in EBdic[EB]['Band']]
                text_sd = 'casa -c ' + SCR_DIR + 'mergeEB.py -u '
                for prefix in splitBandEBList: text_sd = text_sd + prefix + ','
                print(text_sd[:-1])
                os.system(text_sd[:-1])
                execGridReduction('%s.%s' % (newPrefix, bandName))
                text_sd = 'casa -c ' + SCR_DIR + 'splitFluxLog.py -u '
                for prefix in bandEBList: text_sd = text_sd + prefix.split('.')[0] + ','
                print(text_sd[:-1])
                os.system(text_sd[:-1])
                for prefix in splitBandEBList:
                    storeLog(prefix.split('.')[0], bandName)
                    text_sd = replaceList(prefix.split('.')[0], 'done')
                    os.system(text_sd)
                    os.system('mv ./UID/UIDnew ./UID/UIDList')
                    os.system('casa -c ' + SCR_DIR + 'AMAPOLA2SC.py -u %s' % (prefix.split('.')[0]))
            #
        #
        os.system('rm -rf *.ms*')
#
