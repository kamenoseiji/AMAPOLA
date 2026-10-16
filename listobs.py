from optparse import OptionParser
parser = OptionParser()
parser.add_option('-u', dest='prefix', metavar='prefix',
    help='EB UID   e.g. uid___A002_X10dadb6_X18e6', default='')
#
(options, args) = parser.parse_args()
#-------- BB_spw : BB power measurements for list of spws, single antenna, single scan
prefix  = options.prefix.replace("/", "_").replace(":","_").replace(" ","")
importasdm(prefix)
listobs(prefix+'.ms', scan='', spw='', verbose=True, listfile = prefix+'.listobs')
