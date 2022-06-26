## This script benchmarks MMAGPU against MMA1Bit with both mma and wmma.
# We assume a population with 10k individuals.

library(miraculix)
# Global RFutils options
RFoptions(install="none", centered=FALSE, normalized=FALSE, cores=12, helpinfo=FALSE, la_mode=LA_GPU)

# Number of individuals
n <- 10000
# Number of SNPs
snps <- c(seq(1e3, 9e3, by = 1e3), seq(1e4, 9e4, by = 1e4), seq(1e5, 1e6, by = 1e5))
# Matrices for time measurements
snps_time <- matrix(0, nrow=length(snps), ncol=3)
rownames(snps_time) <- snps
colnames(snps_time) <- c("MMAGPU", "mma-MMA1Bit", "wmma-MMA1Bit")
average_benchmark <- 20
divisor <- 1000

# Iterate over number of SNPs
for(i in 1:length(snps)){
  # Simulate subpopulation
  SNPs <- matrix(sample(0:2, n/divisor * snps[i], replace=T), ncol=n/divisor)

  cat("snpcoding=MMAGPU\n")
  # Simulate population
  RFoptions(snpcoding=MMAGPU)
  Z <- miraculix::genomicmatrix(snps[i], n)
  for (j in 0:(divisor-1)) {
    fillGeno(Z, (1:(n/divisor)) + n/divisor * j, SNPs)
  }
  # Start time measurements
  for (j in 1:average_benchmark) {
    Sys.sleep(0.1)
    snps_time[i,1] <- snps_time[i,1] + system.time({ # calculate relationship matrix
      G <- miraculix::relationshipMatrix(Z,
        n_streams = 6, shape = 1, tilesize = 1024
      )
    })[3]
  }
  snps_time[i,1] <- snps_time[i,1]/average_benchmark
  
  cat("snpcoding=MMA1Bit, mma version\n")
  # Simulate population
  RFoptions(snpcoding=MMA1Bit)
  Z <- miraculix::genomicmatrix(snps[i], n)
  for (j in 0:(divisor-1)) {
    fillGeno(Z, (1:(n/divisor)) + n/divisor * j, SNPs)
  }
  # Start time measurements
  for (j in 1:average_benchmark) {
    Sys.sleep(0.1)
    snps_time[i,2] <- snps_time[i,2] + system.time({ # calculate relationship matrix
      G <- miraculix::relationshipMatrix(Z,
        warp = FALSE, shape = 6, n_streams = 6, tilesize = 1536, naive = TRUE
      )
    })[3]
  }
  snps_time[i,2] <- snps_time[i,2]/average_benchmark

  cat("snpcoding=MMA1Bit, wmma version\n")
  # Start time measurements
  for (j in 1:average_benchmark) {
    Sys.sleep(0.1)
    snps_time[i,3] <- snps_time[i,3] + system.time({ # calculate relationship matrix
      G <- miraculix::relationshipMatrix(Z,
        warp = TRUE, shape = 6, n_streams = 6, tilesize = 1024, naive = TRUE
      )
    })[3]
  }
  snps_time[i,3] <- snps_time[i,3]/average_benchmark
  print(snps[i])
}

saveRDS(snps_time, "snps_time.rds")
