#!/usr/bin/env bash
set -e  # stop if any command fails


OUTDIR="output/HealthyLeaf_Schannel-Spunta"
mkdir -p "$OUTDIR"

index1=0

for i in {0..56}; do
    for file in /mnt/c/Users/polyx/Desktop/Github\ Repos/Pattern_spectra_ML/dataset/Spunta/leaf_images/RGBFilter/channels_pgm/${i}_healthybox_Spatch_*.pgm; do
        base=$(basename "$file" .pgm)
        ./xmaxtree "$file" a 9, 0 dl 1, 1 dh 5, 50 m 2, 0 n 10, 10 f 3 nogui e "$OUTDIR/hS_${index1}_${base}"
        index1=$((index1 + 1))
    done
done