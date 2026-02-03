- segment_images.ipynb
- run xmaxtree for the image channels in the extracted folder, using my folder path an example command I used for the R channel  - index1=0; for i in {0..56}; do for file in  /mnt/c/Users/anush/Documents/PostDoc/"Croptimal datasets"/NAKFielddataset/Spunta_variety/leaf_images/RGBFilter/channels_pgm/${i}_healthybox_Rpatch_*.pgm; do base=$(basename "$file" .pgm); ./xmaxtree "$file" a 9, 0 dl 1, 1 dh 5, 50 m 2, 0 n 10, 10 f 3 nogui e "$OUTDIR/hR_${index1}_${base}"; index1=$((index1 + 1)); done; done
- run leaf_classification.py


- For code concerning feature extraction with respect to color spaces and pattern spectra bins, they can be found in the *Feature_Extraction* folder. Morover, SLURM files to run large experiments for bin analysis and colorspace analysis are also available as *run_bin_analysis_job.sh* and *run_colorspace_job.sh* 


