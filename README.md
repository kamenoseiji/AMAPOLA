# AMAPOLA
This repository records scripts for Analytic Matrix for ALMA POLArimetry (AMAPOLA) reduction. AMAPOLA is a set of CASA-friendly Python scripts to reduce Grid Survey datasets to measure full Stokes parameters of calibrators. The results are ingested to the ALMA Calibrator Source Catalogue https://sourcecat.osf.alma.cl/sc/ (only Pol. Angle and Pol. Fraction columns). Full Stokes data are presented in the AMAPOLA website https://www.alma.cl/~skameno/AMAPOLA/.
AMAPOLA system is composed by two parts; data reduction (this repository) and data compilation (another repository https://github.com/kamenoseiji/ALMAR).

## Requirements
- Python (version 3) scripts run under Common Astronomy Software Applications package (CASA : https://casa.nrao.edu/) version 6 or later.
- polQuery.R requires R statistical language version 4.4 or later. RCurl package must be installed, in advance.
- Codes will run standalone.
- Network access (with permission) is required to ingest results onto the source catalogue.
- Further privilege is required to fetch shiftlog and to download ASDMs from the archive. For this purpose, the script checkUID.py is supposed to run under the JAO EOC cluster.

## Installation and setup
1. Define a directory to store the scripts (e.g. /home/skameno/AMAPOLA/ )

2. At the directory (e.g. /home/skameno/), run
   ``` bash
   git clone https://github.com/kamenoseiji/AMAPOLA.git
   ```
   this spawns /home/skameno/AMAPOLA/ where all scripts will be stored.

3. Add the directory on the PYTHONPATH environment in .bashrc:
   ``` .bashrc
   export PYTHONPATH=${PYTHONPATH}:/home/skameno/AMAPOLA/ 
   ```
   then, source ~/.bashrc to activate the environment parameter.

## How it works
### For individual MS
 If you already have a measurement set (MS) generated from an ASDM at a working directory, run 
   ``` CASA
   casa -c checkGridSurvey.py -u [UID]
   e.g. casa -c ~/AMAPOLA/checkGridSurvey.py -u uid___A002_X1400c05_X1f4c
   ```
   Note that both ASDM and MS must exist on the same working directory.

### For multiple ASDMs
 Automated reduction can be performed by a wrapper script 'InfLoop.sh'
   ``` InfLoop.sh
   !/usr/bin/bash
   rm -rf uid*
   while true
   do
        python ~/AMAPOLA/SLTquery.py -n Grid_ > UID/UIDList
        python ~/AMAPOLA/SLTquery.py -s CS_Flux >> UID/UIDList
        python ~/AMAPOLA/SLTquery.py -s CS_Struct >> UID/UIDList
        python ~/AMAPOLA/SLTquery.py -s Mini_All_Sky >> UID/UIDList
        python ~/AMAPOLA/SLTquery.py -b Baseline >> UID/UIDList
        casa -c ~/AMAPOLA/checkUID.py
        sleep 21600
    done
   ```
 This script activates the procedure every 6 hours (21600 seconds). SLTquery.py fetch new ExecBlock records in the shiftlog and records them on UID/UIDList. For this purpose, a subdirectory UID must be created, in advance, undr the working directory. Then, the script checkUID.py runs to download ASDMs, merge multiple EBs with the same array and band combination, executes checkGridSurvey.py, and ingest results on the source catalogue.
 These subdirectories must be created with write permission before running InfLoop.sh:
- UID/ : to store UIDList and ASDMs
- PDF/ : to store plots
- NPY/ : to store numpy array files
- LOG/ : to store reduction logs
