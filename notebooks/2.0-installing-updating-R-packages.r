# IMPORTING REQUESTED PACKAGES
# NEED TO EXECUTE ONLY ONE TIME, AFTER INCLUDED INTO ENVIRONMENT NO NEED TO INSTALL PACLAGES AGAIN

install.packages("BiocManager", dependencies = TRUE)

# After install BiocManager, restart R kernel and install other packages if they do not exist in your environment

BiocManager::install("hipathia", dependencies = TRUE) # hipathia package
BiocManager::install("org.Mm.eg.db", dependencies = TRUE) #mmu, converting entrez to gene symbol
BiocManager::install("AnnotationDbi", dependencies = TRUE)
install.packages('igraph') # converting graph to graphml format

"OPTIONAL -- UPDATE OLD PACKAGES"
# list all packages where an update is available
length(old.packages())
# print(length(old.packages()))

# update, without prompts for permission/clarification
# update.packages(ask = FALSE)
# tools::package_dependencies("BiocManager", db = installed.packages())Ç
# install.packages("AnnotationDbi", version = "1.49")