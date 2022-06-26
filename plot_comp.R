## This script provides graphics and tables for the bachelor thesis
library(tidyverse)
library(ggplot2)
library(reshape2)
library(kableExtra)

snps <- readRDS("snps_time.rds")
plot1 <- snps %>% melt() %>%
  ggplot() + 
  theme_minimal() + 
  theme(legend.position=c(.85,.25), legend.text=element_text(size=10)) +
  labs(#title=expression(paste(paste("Zeitmessung für die Berechnung von ", Z*Z^T), " bei 10000 Individuen")),
        x ="Anzahl SNPs", y = "Zeit in Sekunden") +
  geom_line(aes(x=Var1, y=value, col=Var2)) + 
  theme(legend.title=element_blank()) +
  scale_y_continuous(labels=function(x) format(x, decimal.mark = ",", scientific = FALSE))
print(plot1)
ggsave(file="snps.pdf", width = 400/2, height = 230/2, units = "mm")

indivs <- readRDS("indivs_time.rds")
plot2 <- indivs %>% melt() %>%
  ggplot() + 
  theme_minimal() +
  theme(legend.position=c(.85,.25), legend.text=element_text(size=10)) +
  labs(#title=expression(paste(paste("Zeitmessung für die Berechnung von ", Z*Z^T), " bei 10000 Individuen")),
    x ="Anzahl Individuen", y = "Zeit in Sekunden") +
  geom_line(aes(x=Var1, y=value, col=Var2)) + 
  theme(legend.title=element_blank()) +
  scale_y_continuous(labels=function(x) format(x, decimal.mark = ",", scientific = FALSE))
print(plot2)
ggsave(file="indivs.pdf", width = 400/2, height = 230/2, units = "mm")

  shapes <- c(
  "64x64x512_32x32x512_75",
  "64x128x512_32x64x512_75",
  "128x64x512_64x32x512_75",
  "64x256x512_64x64x512_75",
  "256x64x512_64x64x512_75",
  "128x128x512_64x64x512_75",
  "128x256x512_64x64x512_75",
  "256x128x512_64x64x512_75",
  "64x128x1024_32x64x1024_80",
  "128x64x1024_64x32x1024_80",
  "64x64x1024_32x32x1024_80",
  "64x64x512_32x32x512_80",
  "64x128x512_32x64x512_80",
  "128x64x512_64x32x512_80",
  "64x256x512_64x64x512_80",
  "256x64x512_64x64x512_80",
  "128x128x512_64x64x512_80",
  "128x256x512_64x64x512_80",
  "256x128x512_64x64x512_80"
)

tile_time <- readRDS("tile_time.rds")[,,1] %>% round(digits = 3)
rownames(tile_time) <- shapes
tile_time[,1:ncol(tile_time)] %>%
  kbl(
    caption="Vergleich verschiedener Tilesizes und GEMM-shapes",
    format="latex",
    col.names = seq(512, by=256, to=2048),
    align="r"
  ) %>%
  kable_minimal(full_width = F)

shapes_old <- c(
  "128x256x128_64x64x128",
  "256x128x128_64x64x128",
  "128x128x128_64x64x128",
  "64x128x128_32x64x128",
  "128x64x128_64x32x128",
  "64x64x128_32x32x128"
)

tile_time_old <- readRDS("tile_time_old.rds")[,,1] %>% round(digits = 3)
rownames(tile_time_old) <- shapes_old
tile_time_old[,1:ncol(tile_time_old)] %>%
  kbl(
    caption="Vergleich verschiedener Tilesizes und GEMM-shapes",
    format="latex",
    col.names = seq(512, by=256, to=2048),
    align="r"
  ) %>%
  kable_minimal(full_width = F)

shapes_wmma <- c(
  "64x64x512_32x32x512_75",
  "64x128x512_32x64x512_75",
  "128x64x512_64x32x512_75",
  "64x256x512_64x64x512_75",
  "256x64x512_64x64x512_75",
  "128x128x512_64x64x512_75",
  "128x256x512_64x64x512_75",
  "256x128x512_64x64x512_75"
)

tile_time_wmma <- readRDS("tile_time_wmma.rds")[,,1] %>% round(digits = 3)
rownames(tile_time_wmma) <- shapes_wmma
tile_time_wmma[,1:ncol(tile_time_wmma)] %>%
  kbl(
    caption="Vergleich verschiedener Tilesizes und GEMM-shapes",
    format="latex",
    col.names = seq(512, by=256, to=2048),
    align="r"
  ) %>%
  kable_minimal(full_width = F)

my.min <- function(x) ifelse(!all(is.na(x)), min(x, na.rm=T), NA)
printf <- function(...) invisible(print(sprintf(...)))
# get best time
best <- which(tile_time == my.min(tile_time), arr.ind = TRUE)
print("Minimum MMA1Bit (mma)")
print(best)
best <- which(tile_time_wmma == my.min(tile_time_wmma), arr.ind = TRUE)
print("Minimum MMA1Bit (wmma)")
print(best)
best <- which(tile_time_old == my.min(tile_time_old), arr.ind = TRUE)
print("Minimum MMAGPU")
print(best)
# print(my.min(tile_time))


