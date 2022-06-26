### This script is a collection of profiling commands for the CUDA part of miraculix.

## Used in Bachelor Thesis
sudo -E ncu -o ./profiling --force-overwrite --target-processes all --replay-mode application --app-replay-match grid --app-replay-buffer file --kernel-name-base function --launch-skip-before-match 0 --section LaunchStats --section Occupancy --section SpeedOfLight --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --call-stack --profile-from-start 1 --cache-control none --clock-control base --apply-rules no --import-source no --check-exit-code no Rscript profiling.R
sudo -E ncu -o ./profiling_wmma --force-overwrite --target-processes all --replay-mode application --app-replay-match grid --app-replay-buffer file --kernel-name-base function --launch-skip-before-match 0 --section LaunchStats --section Occupancy --section SpeedOfLight --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --call-stack --profile-from-start 1 --cache-control none --clock-control base --apply-rules no --import-source no --check-exit-code no Rscript profiling_wmma.R
sudo -E ncu -o ./profiling_old --force-overwrite --target-processes all --replay-mode application --app-replay-match grid --app-replay-buffer file --kernel-name-base function --launch-skip-before-match 0 --section LaunchStats --section Occupancy --section SpeedOfLight --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --call-stack --profile-from-start 1 --cache-control none --clock-control base --apply-rules no --import-source no --check-exit-code no Rscript profiling_old.R

## For MemoryWorkloadAnalysis
# sudo -E ncu -o profile -f --section MemoryWorkloadAnalysis --target-processes all Rscript profiling.R

## Open files with
# ncu-ui profiling.ncu-rep

## valgrind for memory leak analysis
# R -d "valgrind --target-processes all" -f profiling.R

## deprecated tools -- don't work with 8.0 or higher
# nvprof --profile-child-processes -o test%p.prof Rscript profiling.R
# nvvp -vm /usr/lib/jvm/jdk1.8.0_301/bin -o test%p.prof

## Help for problems with Java version of nvvp
# https://forums.developer.nvidia.com/t/cannot-launch-nvidia-visual-profiler/48214/28?page=2
# https://www.oracle.com/java/technologies/javase/javase8u211-later-archive-downloads.html
