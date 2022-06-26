## This script compares different tilesizes, number of streams and gemm shapes for MMA1Bit. 
# We assume a population with 10k individuals and 1000k SNPs.

library(miraculix)
RFoptions(install="none")
RFoptions(centered=FALSE, normalized=FALSE)

# Number of individuals
n <- 1024
snps <- 1000448 # to be divisible by 256
# Global RFutils options
RFoptions(cores=12,helpinfo=FALSE)
# Matrices for time measurements
# tiles <- seq(512, by=256, to=1536)
# streams <- seq(4, by=1, to=12)
# shapes <- 0:7
tiles <- c(1024)
streams <- c(6)
shapes <- c(6)
tile_time <- array(NA, c(length(shapes), length(tiles), length(streams)))
average_matrix <- 1
divisor <- 64

# Simulate population
SNPs <- matrix(sample(1, n/divisor * snps, replace=T), ncol=n/divisor)

# MMA1Bit options
RFoptions(la_mode=LA_GPU,snpcoding=MMA1Bit)
cat("snpcoding=MMA1Bit, mma version\n")

Z <- miraculix::genomicmatrix(snps, n)
for (j in 0:(divisor-1)) {
  fillGeno(Z, (1:(n/divisor)) + n/divisor * j, SNPs)
}

for (shape in 1:length(shapes)) {
  for (tile in 1:length(tiles)) {
    print(tile)
    for (stream in 1:length(streams)) {
      # limit global memory to value necessary on my machine (RTX 2070)
      if (streams[stream]*(tiles[tile]^2*4 + (snps/2) * tiles[tile]) <= 8361213952*0.8) {
        tile_time[shape,tile,stream] <- 0
        for (i in 1:average_matrix) {
          Sys.sleep(0.2) # slight time buffer
          # Berechnung Verwandtschaftsmatrix
          tile_time[shape,tile,stream] <- tile_time[shape,tile,stream] + system.time({
            G <- miraculix::relationshipMatrix(Z,
              warp = TRUE, shape = shapes[shape], n_streams = streams[stream], tilesize = tiles[tile], naive = FALSE
            )
          })[3]
          print(G[1:10,1:10])
          print(G[1015:1024,1015:1024])
          # print(match(0, G))
          # system.time({
          #   G <- miraculix::relationshipMatrix(Z,
          #     warp = TRUE, shape = shapes[shape], n_streams = streams[stream], tilesize = tiles[tile], naive = TRUE
          #   )
          # })[3]
          # print(G[1:10,1:10])
          # print(G[1015:1024,1015:1024])
          # print(match(0, G))
        }
        tile_time[shape,tile,stream] <- tile_time[shape,tile,stream]/average_matrix
      }
    }
  }
}

# RFoptions(la_mode=LA_GPU,snpcoding=MMAGPU)
# cat("snpcoding=MMA1Bit, mma version\n")

# Z <- miraculix::genomicmatrix(snps, n)
# for (j in 0:(divisor-1)) {
#   fillGeno(Z, (1:(n/divisor)) + n/divisor * j, SNPs)
# }

# for (shape in 1:length(shapes)) {
#   for (tile in 1:length(tiles)) {
#     print(tile)
#     for (stream in 1:length(streams)) {
#       # limit global memory to value necessary on my machine (RTX 2070)
#       if (streams[stream]*(tiles[tile]^2*4 + (snps/2) * tiles[tile]) <= 8361213952*0.8) {
#         tile_time[shape,tile,stream] <- 0
#         for (i in 1:average_matrix) {
#           Sys.sleep(0.2) # slight time buffer
#           # Berechnung Verwandtschaftsmatrix
#           tile_time[shape,tile,stream] <- tile_time[shape,tile,stream] + system.time({
#             G <- miraculix::relationshipMatrix(Z,
#               warp = TRUE, shape = shapes[shape], n_streams = streams[stream], tilesize = tiles[tile], naive = TRUE
#             )
#           })[3]
#           print(G[1:10,1:10])
#           print(G[1015:1024,1015:1024])
#         }
#         tile_time[shape,tile,stream] <- tile_time[shape,tile,stream]/average_matrix
#       }
#     }
#   }
# }

# helper functions
my.min <- function(x) ifelse(!all(is.na(x)), min(x, na.rm=T), NA)
printf <- function(...) invisible(print(sprintf(...)))
# get best time
best <- which(tile_time == my.min(tile_time), arr.ind = TRUE)
print(my.min(tile_time))
printf("shape: %i tilesize: %i streams: %i", shapes[best[1]], tiles[best[2]], streams[best[3]])
rm(Z)
saveRDS(tile_time, "tile_time.rds")
