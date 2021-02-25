# demo.R
# Author: Minzhe Guo (minzhe.guo@cchmc.org)
# Description: 
#				A demonstration of using SINCERA to analyse E16.5 mouse lung single cell data.
# 				The data are RNA-seq of 148 single cells from two independent sample preparations 
#               (86 cells from sample 1 and 62 cells from sample 2) from E16.5 mouse lung and RNA
#               expression values were calculated using the FPKM.
#				Through the pipeline analysis, we distinguished major cell types of fetal mouse lung, 
#               including epithelial, endothelial, smooth muscle, pericyte, and fibroblast-like cell types, 
#               and identified cell type specific gene signatures, bioprocesses, and key regulators.
# Dependencies: 
#             SINCERA a10142015
#             E16.5.Rda
# Date: October 14, 2015

# ============================= 


#######################################################
#             Load Sincera Code and E16.5 Data        #
#######################################################

#NOTE: please make sure that demo.R, sincera.R, and E16.5.Rda are placed in the same folder and the folder is the current working directory of R

source("sincera.R")
#load("E16.5.Rda")

#head(cells)

#head(genes)

#expressions[1:5,1:5]

args<-commandArgs(trailingOnly=TRUE)
#indata=args[1]
#incell=args[2]
#args = commandArgs(trailingOnly=TRUE)
vct=as.numeric(args[1])
clust=args[2]
indata=args[3]
#incell=args[3]
outfile=args[4]
myexpressions <-read.table(indata, sep='\t', header = T,row.names=1)
#mycells <-read.table(incell, sep='\t', header = T)
#Input <- t(b) # data matrix, cells in rows, genes in columns
#print(myexpressions[1:5,1:5])
#print(mycells[1:5,])
#print(expressions[1:5,1:5])
#print(cells[1:5,])

expressions=myexpressions
#cells=mycells
print(expressions[1:5,1:5])
#print(cells[1:5,])
#expressions <-read.table(indata, sep='\t', header = T,row.names=1,check.names=FALSE)
#cells <-read.table(incell, sep='\t', header = T,row.names=1)
#print(expressions[1:5,1:5] )
#print(cells[1:5,:])
#pd <- new("AnnotatedDataFrame", data = cells)

ES <- new("ExpressionSet", exprs = as.matrix(expressions))
#ES <- new("ExpressionSet", exprs = as.matrix(expressions), phenoData = pd)


#ES <- exprs.truncate(ES, upper.bound=NULL, lower.bound=0.01)


#ESz <- normalization.zscore(ES, group.by="SAMPLE", groups=NULL, verbose=TRUE) 
if(clust=='hc'){
	ret <- cluster.assignment(ES, cluster.label="CLUSTER", 
			       clustering.method="hc", 
			       distance.method="pearson", #"spearman","euclidean" 
			       linkage.method="average", 
			       num.singleton=0, h=0.5, k=vct, 
			       do.shift=TRUE, do.plot=TRUE, 
			       verbose=TRUE, export=TRUE, export.components="pd")
	clusters <- ret$cell.cluster
	print(clusters)
}
# the cluster membership shall be encoded in ret$cell.cluster

# consensus clustering
#notrun  ret <- cluster.assignment(ESz, cluster.label="consensus_cluster", clustering.method="consensus", distance.method="pearson", linkage.method="average", min.area.increase=0.2, maxK=10, reps=10, pItem=0.8, pFeature=1, clusterAlg="hc",verbose=TRUE, export=TRUE, export.components="pd")

if(clust=='consensus'){
	library(ConsensusClusterPlus)
	fpkm <- exprs(ES) 
	ret <- ConsensusClusterPlus(fpkm, maxK=vct, reps=10, pItem=0.8, pFeature=1, clusterAlg='hc', innerLinkage='average',finalLinkage='average', distance='pearson')
        clusters <- as.character(ret[[vct]][["consensusClass"]])
	names(clusters) <- colnames(exprs(ES))
	print(clusters)
}
#tight clustering
#ret <- cluster.assignment(ES, cluster.label="tight_cluster", clustering.method="tight", target=vct, k.min=vct+5, verbose=TRUE)
if(clust=='tightclust'){
	library(tightClust)
	fpkm <- exprs(ES) 
	ret <- tight.clust(t(fpkm), target=vct-1, k.min=vct+4)
	clusters <- ret$cluster
	names(clusters) <- colnames(exprs(ES))
	print(clusters)
}
#ret <- cluster.assignment(ES, cluster.label="tight_cluster", clustering.method="tight", target=vct, k.min=vct+5, verbose=TRUE, export=TRUE, export.components="pd")

# if hc is used, cell order in the dendrogram shall be encoded in ret$cell.order
#cell_order <- ret$cell.order
#print(cell_order)
#cell_cluster <- ret$cell.cluster
#print(cell_cluster)
write(clusters,file=outfile,sep='\n')
