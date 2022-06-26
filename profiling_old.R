## This script is for ncu-profiling MMAGPU.
# We simulate a population with 1000k SNPs and 10k individuals.

library(miraculix)
# Global RFutils options
RFoptions(install="none", warn_parallel=FALSE)
RFoptions(centered=FALSE, normalized=FALSE, cores=12, helpinfo=FALSE, la_mode=LA_GPU, snpcoding=MMAGPU)

# Number of individuals
n <- 10000
# Number of SNPs
snps <- 1e6
# magic number to avoid RAM overflow
divisor <- 100

#Simulate population
SNPs <- matrix(sample(0:2, n/divisor * snps, replace=T), ncol=n/divisor)
Z <- miraculix::genomicmatrix(snps, n)
for (j in 0:(divisor-1)) {
  fillGeno(Z, (1:(n/divisor)) + n/divisor * j, SNPs)
}

cat("snpcoding=MMAGPU\n")

print(system.time({
    G <- miraculix::relationshipMatrix(Z,
      n_streams = 6, tilesize = 1024
    )
  })[3]
)
