#########################################################################
#Libraries
#########################################################################

# Rscript aglo_cluster_nodes.r -nodes path/to/nodes.npy -odir output_directory

options(warn = -1)
library(argparse)
library(RcppCNPy)
library(ggplot2)

##########################################################################
#Argument Parsing
##########################################################################

parser <- ArgumentParser(description = "Clustering aglomerativo de nodos")
parser$add_argument('-nodes', "--nodes", type = "character", help = "Ruta al archivo .npy de la matriz de nodos", required = TRUE)
parser$add_argument('-odir', "--outdir", type = "character", default = "resultados", help = "Directorio de salida")
args <- parser$parse_args()

nodes_path <- args$nodes   
if (!file.exists(nodes_path)) {
  stop("Nodes file does not exist: ", nodes_path)
}

output_dir <- args$outdir        
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}


############################################################################
# Aglomerative Clustering
############################################################################

flow_weight <- npyLoad(nodes_path)

cat('Matriz original\n')
cat(flow_weight)
cat('\n\n')

#flow_weight <- as.matrix(flow_weight)
flow_weight <- t(flow_weight)

cat('Matriz transpuesta\n')
cat(flow_weight)

#flow_weight <- flow_weight[, apply(flow_weight, 2, function(col) all(is.finite(col))), drop = FALSE]
#
## Eliminar columnas que sean todo ceros
#flow_weight <- flow_weight[, apply(flow_weight, 2, function(col) any(col != 0)), drop = FALSE]
#
## Verifica dimensiones finales
#cat("Dimensiones de flow_weight después de limpieza:\n")
#print(dim(flow_weight))

distance_matrix <- dist(flow_weight, method = "euclidean")
saveRDS(distance_matrix, file = file.path(output_dir, "distance_matrix.rds"))


hc <- hclust(distance_matrix, method = "complete")
dhc <- as.dendrogram(hc)

#############################################################################
# Save clustering results
#############################################################################

n_ <- ncol(flow_weight)
for (i in 1:n_) {
  cluster <- cutree(hc, k = i)
  ruta_etiquetas <- file.path(output_dir, paste0("labels_k", i, ".npy"))
  npySave(ruta_etiquetas, cluster)
}

pdf(file = file.path(output_dir, "dendrogram.pdf"), width = 18, height = 12)

plot(dhc, main = "Dendrogram of Agglomerative Clustering", xlab = "", ylab = "")

ks <- 2:(n_ - 1)
n <- length(hc$order)
heights_k <- hc$height[ (n - ks)]

for (h in heights_k) {
  abline(h = h, col = "red", lty = 3)
}

axis(side = 4, at = heights_k, labels = ks, las = 1, col.axis = "blue", cex.axis = 0.8)
mtext("k", side = 4, line = 3, col = "blue", cex = 0.9)

invisible(dev.off())

pdf(file = file.path(output_dir, "hierarchical.pdf"), width = 18, height = 12)

plot(hc, main = "Hierarchical Clustering", xlab = "", ylab = "")

for (h in heights_k) {
  abline(h = h, col = "red", lty = 3)
}

axis(side = 4, at = heights_k, labels = ks, las = 1, col.axis = "blue", cex.axis = 0.8)
mtext("k", side = 4, line = 3, col = "blue", cex = 0.9)

invisible(dev.off())
cat('Clustering completed!\n')