## This script is checking whether the MMA1Bit functions are working as intended.
# We simulate a population with 100k SNPs and 10k individuals.

library(miraculix)
# Global RFutils options
RFoptions(install="none", warn_parallel=FALSE)
RFoptions(centered=FALSE, normalized=FALSE, cores=12, helpinfo=FALSE, la_mode=LA_GPU)

# Number of individuals
n <- 10000
# Number of SNPs
snps <- 1e6
# magic number to avoid RAM overflow
divisor <- 1000

# Simulate population
SNPs <- matrix(sample(0:2, n/divisor * snps, replace=T), ncol=n/divisor)

# MMAGPU (reference)
RFoptions(snpcoding=MMAGPU)
Z <- miraculix::genomicmatrix(snps, n)
for (j in 0:(divisor-1)) {
  fillGeno(Z, (1:(n/divisor)) + n/divisor * j, SNPs)
}

cat("snpcoding=MMA1Bit, mma version\n")

G <- miraculix::relationshipMatrix(Z,
  n_streams = 6, shape = 1, tilesize = 1024
)

# MMA1Bit (naive)
RFoptions(snpcoding=MMA1Bit)
Z <- miraculix::genomicmatrix(snps, n)
for (j in 0:(divisor-1)) {
  fillGeno(Z, (1:(n/divisor)) + n/divisor * j, SNPs)
}

H <- miraculix::relationshipMatrix(Z,
  warp = FALSE, shape = 6, n_streams = 6, tilesize = 1024, naive = TRUE
)

# Sanity checks
if (all(G == H)) {
  cat("\nPass.\n\n")
} else {
  cat("\nWrong values!\n\n")
  print(G[1:10,1:10])
  print(H[1:10,1:10])
  print(sum(G!=H))
  print(match(1, G!=H))
}

# MMA1Bit (naive = FALSE)
RFoptions(snpcoding=MMA1Bit)
Z <- miraculix::genomicmatrix(snps, n)
for (j in 0:(divisor-1)) {
  fillGeno(Z, (1:(n/divisor)) + n/divisor * j, SNPs)
}

H <- miraculix::relationshipMatrix(Z,
  warp = FALSE, shape = 6, n_streams = 6, tilesize = 1024, naive = FALSE
)

# Sanity checks
if (all(G == H)) {
  cat("\nPass.\n\n")
} else {
  cat("\nWrong values!\n\n")
  print(G[1:10,1:10])
  print(H[1:10,1:10])
  print(sum(G!=H))
  print(match(1, G!=H))
}