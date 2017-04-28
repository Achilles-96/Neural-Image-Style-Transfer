content="$1"
style="$2"
python createpyramid.py $1
CUDA_VISIBLE_DEVICES=1 th neuralpyramid.lua $1 $2
CUDA_VISIBLE_DEVICES=1 th neural.lua "InputContentImages/${content}.jpg" "pyramid_${content}_${style}.jpg" "pyramid_${content}_${style}_final.jpg"