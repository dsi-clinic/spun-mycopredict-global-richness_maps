gdalwarp \
  -overwrite \
  -multi \
  -wo NUM_THREADS=ALL_CPUS \
  -srcnodata nan \
  -dstnodata nan \
  -te -180 -90 180 90 \
  -tr 0.008333333333333 0.008333333333333 \
  -tap \
  -ot Float32 \
  -co TILED=YES \
  -co COMPRESS=LZW \
  -co PREDICTOR=3 \
  -co BIGTIFF=YES \
  output/original24-world/tiles/*.tif \
  output/original24-world.tif
