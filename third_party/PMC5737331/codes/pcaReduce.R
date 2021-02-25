args = commandArgs(trailingOnly=TRUE)
vct=as.numeric(args[1])
indata=args[2]
inans=args[3]
outfile=args[4]
b <-read.table(indata, sep='\t', header = T,row.names=1)
ans <-read.table(inans, sep='\t', header = F)
Input <- t(b) # data matrix, cells in rows, genes in columns
if(T){
	library("pcaReduce")
	Output_S <- PCAreduce(Input, nbt=1, q=vct*3, method='M')
	res=Output_S[[1]][,vct*2+2]
	write(res,file=outfile,sep='\n')
}
