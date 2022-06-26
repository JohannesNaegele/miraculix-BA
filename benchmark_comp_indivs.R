## This script benchmarks MMAGPU against MMA1Bit with both mma and wmma.
# We assume a population with 1000k snps.

library(miraculix)
# Global RFutils options
RFoptions(install="none", centered=FALSE, normalized=FALSE, cores=12, helpinfo=FALSE, la_mode=LA_GPU)

# Number of individuals
n <- seq(1000, 10000, by = 1000)
# Number of SNPs
snps <- 1e6
# Matrices for time measurements
indivs_time <- matrix(0, nrow=length(n), ncol=3)
rownames(indivs_time) <- n
colnames(indivs_time) <- c("MMAGPU", "mma-MMA1Bit", "wmma-MMA1Bit")
average_benchmark <- 20
divisor <- 1000

# Iterate over number of SNPs
for(i in 1:length(n)){
  # Simulate subpopulation
  SNPs <- matrix(sample(0:2, n[i]/divisor * snps, replace=T), ncol=n[i]/divisor)

  cat("snpcoding=MMAGPU\n")
  # Simulate population
  RFoptions(snpcoding=MMAGPU)
  Z <- miraculix::genomicmatrix(snps, n[i])
  for (j in 0:(divisor-1)) {
    fillGeno(Z, (1:(n[i]/divisor)) + n[i]/divisor * j, SNPs)
  }
  # Start time measurements
  for (j in 1:average_benchmark) {
    Sys.sleep(0.1)
    indivs_time[i,1] <- indivs_time[i,1] + system.time({ # calculate relationship matrix
      G <- miraculix::relationshipMatrix(Z,
        n_streams = 6, shape = 1, tilesize = 1024
      )
    })[3]
  }
  indivs_time[i,1] <- indivs_time[i,1]/average_benchmark
  
  cat("snpcoding=MMA1Bit, mma version\n")
  # Simulate population
  RFoptions(snpcoding=MMA1Bit)
  Z <- miraculix::genomicmatrix(snps, n[i])
  for (j in 0:(divisor-1)) {
    fillGeno(Z, (1:(n[i]/divisor)) + n[i]/divisor * j, SNPs)
  }
  # Start time measurements
  for (j in 1:average_benchmark) {
    Sys.sleep(0.1)
    indivs_time[i,2] <- indivs_time[i,2] + system.time({ # calculate relationship matrix
      G <- miraculix::relationshipMatrix(Z,
        warp = FALSE, shape = 6, n_streams = 6, tilesize = 1536, naive = TRUE
      )
    })[3]
  }
  indivs_time[i,2] <- indivs_time[i,2]/average_benchmark

  cat("snpcoding=MMA1Bit, wmma version\n")
  # Start time measurements
  for (j in 1:average_benchmark) {
    Sys.sleep(0.1)
    indivs_time[i,3] <- indivs_time[i,3] + system.time({ # calculate relationship matrix
      G <- miraculix::relationshipMatrix(Z,
        warp = TRUE, shape = 6, n_streams = 6, tilesize = 1024, naive = TRUE
      )
    })[3]
  }
  indivs_time[i,3] <- indivs_time[i,3]/average_benchmark
  print(snps)
}

saveRDS(indivs_time, "indivs_time.rds")
