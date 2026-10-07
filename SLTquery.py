import sys
import os
import datetime
from optparse import OptionParser
parser = OptionParser()
parser.add_option('-s', dest='SBcode', metavar='SBcode',
    help='SB code  e.g. PSO_Grid', default='')
parser.add_option('-b', dest='Subject', metavar='Subject',
    help='Subject  e.g. baseline', default='')
parser.add_option('-n', dest='ProjectName', metavar='ProjectName',
    help='ProjectName  e.g. Per Source Grid Monitoring', default='')
parser.add_option('-d', dest='days', metavar='days',
    help='Backword days', default='4')
(options, args) = parser.parse_args()
backDays=  int(options.days)
UserPass = os.getenv('b64credentials', '')   # Authorization Basic Base64
if UserPass == '':
    print('Set b64credentials as an environmental variable')
    exit()
SLT_URI  = 'https://asa.alma.cl/webslt/service/api/entries?'
ST = list(range(0, backDays, 100))
ET = [st-1 for st in ST[1:]] + [backDays]
for term_index, term in enumerate(ET):
    SLTstart = (datetime.datetime.today() - datetime.timedelta(days=term) - datetime.timedelta(hours=4)).strftime('%Y-%m-%dT%H:%M:%S')
    SLTend   = (datetime.datetime.today() - datetime.timedelta(days=ST[term_index]) - datetime.timedelta(hours=2)).strftime('%Y-%m-%dT%H:%M:%S')
    queryText = 'curl -H\"Authorization: Basic %s\" \'%sintervalStart=%s&intervalEnd=%s&status=success' % (UserPass, SLT_URI, SLTstart, SLTend)
    if options.SBcode  != '': queryText = queryText + '&schedBlockCode=' + options.SBcode
    if options.Subject != '': queryText = queryText + '&subject=' + options.Subject
    if options.ProjectName != '': queryText = queryText + '&projectName=' + options.ProjectName
    queryText = queryText + '\' > SLT.log'
    #print(queryText)
    os.system(queryText)
    fp = open('SLT.log', 'r')
    SLTline = fp.readlines()
    fp.close()
    SLTentries = SLTline[0].split('\"')
    UIDList = [SLTentries[index+2] for index, SLTentry in enumerate(SLTentries) if 'execBlockUid' in SLTentry]
    arrayNameList = [SLTentries[index+2] for index, SLTentry in enumerate(SLTentries) if 'arrayName' in SLTentry]
    startTimeList = [SLTentries[index+2] for index, SLTentry in enumerate(SLTentries) if 'start' in SLTentry]
    endTimeList   = [SLTentries[index+2] for index, SLTentry in enumerate(SLTentries) if 'end' in SLTentry]
    SBnameList   = [SLTentries[index+2] for index, SLTentry in enumerate(SLTentries) if 'schedBlockCode' in SLTentry]
    for index, UID in enumerate(UIDList): print('new %s %s %s %s %s' % (UID, arrayNameList[index], startTimeList[index], endTimeList[index], SBnameList[index]))
