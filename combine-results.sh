gdalwarp \
  -overwrite \
  -ot Float32 \
  -srcnodata nan \
  -dstnodata nan \
  -co TILED=YES \
  -co COMPRESS=LZW \
  -co BIGTIFF=YES \
  data/alpha_earth_california_2017/raw_tiles/2017/10N/*.tiff \
  output/alphaearth-california-10N.tif

gdalwarp \
  -overwrite \
  -ot Float32 \
  -srcnodata nan \
  -dstnodata nan \
  -co TILED=YES \
  -co COMPRESS=LZW \
  -co BIGTIFF=YES \
  data/alpha_earth_california_2017/raw_tiles/2017/11N/*.tiff \
  output/alphaearth-california-11N.tif

gdalwarp \
  -overwrite \
  -multi \
  -wo NUM_THREADS=ALL_CPUS \
  -r average \
  -t_srs EPSG:4326 \
  -tr 0.008333333333333 0.008333333333333 \
  -tap \
  -srcnodata nan \
  -dstnodata nan \
  -ot Float32 \
  -co TILED=YES \
  -co COMPRESS=LZW \
  -co BIGTIFF=YES \
  data/alpha_earth_california_2017/raw_tiles/2017/10N/*.tiff \
  data/alpha_earth_california_2017/raw_tiles/2017/11N/*.tiff \
  output/alphaearth-california-worldgrid.tif
