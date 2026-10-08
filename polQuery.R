library(RCurl)
eval(parse(text = getURL("https://raw.githubusercontent.com/kamenoseiji/ALMAR/refs/heads/master/StatStokes.R", ssl.verifypeer = FALSE)))
#source('~/ALMAR/StatStokes.R')
Arguments <- commandArgs(trailingOnly = T)
timeWindow <- 30* 86400
#Arguments <- c('-D2026/08/26/07:55:56', '-F97.500000', 'J0423-0120', 'J0106-4034', 'J0210-5101', 'J0348-2749', 'J0217+0144', 'J0108+0135', 'J2258-2758', 'J0309-6058', 'J0238+1636', 'J2348-1631', 'J2223-3137', 'J0450-8101', 'J0522-3627', 'J0237+2848', 'J2232+1143')
#-------- Parse arguments
parseArg <- function( args ){
	srcNum <- argNum <- length(args)
	for( index in 1:argNum ){
		if(substr(args[index], 1,2) == "-D"){ refDate <- as.POSIXct(substring(args[index], 3)); srcNum <- srcNum - 1}
		if(substr(args[index], 1,2) == "-F"){ refFreq <- as.numeric(substring(args[index], 3)); srcNum <- srcNum - 1}
	}
	srcList = args[(argNum - srcNum + 1):argNum]
	return(list(refDate = refDate, refFreq = refFreq, srcList = srcList[grep('^J[0-9]',srcList )]))
}
argList <- parseArg(Arguments)
refDate <- argList$refDate
refFreq <- argList$refFreq
srcList <- argList$srcList
#-------- Load Flux.Rdata from web
#if( file.exists('Flux.Rdata') == FALSE ){ download.file("https://www.alma.cl/~skameno/AMAPOLA/Flux.Rdata", "Flux.Rdata") }
#load('Flux.Rdata')
load(url("https://www.alma.cl/~skameno/AMAPOLA/Flux.Rdata")) 
#-------- For each source
IQUV <- data.frame(Src='', I=0.0, Q=0.0, U=0.0, V=0.0, P=0.0, EVPA=NA, eI=0.0, eQ=0.0, eU=0.0, eV=0.0, eP=0.0, eEVPA=0.0)
for(sourceName in srcList){
	SDF <- FLDF[((FLDF$Src == sourceName) & (abs(as.numeric(FLDF$Date) - as.numeric(as.POSIXct(refDate))) < timeWindow)),]
    if(nrow(SDF) < 1){ next }
    IQUV <- rbind(IQUV, estimateIQUV(SDF, refFreq, as.POSIXct(refDate)))
}
IQUV <- na.omit(IQUV)
df <- data.frame(Src=IQUV$Src, I=sprintf('%+7.3f',IQUV$I), Q=sprintf('%+6.3f', IQUV$Q), U=sprintf('%+6.3f', IQUV$U))
write.table(df, file="CalQU.data", append=F, quote=F, col.names=F, row.name=F)
