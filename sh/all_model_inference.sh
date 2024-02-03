X=38
Y=35
DEPTH=14.64
MAG=5.0
python ../src/baseline_inference.py -x $X -y $Y -depth $DEPTH -mag $MAG
python ../src/cls_inference.py -g 0 -m ../result/result_20230807_183130/model_best -x $X -y $Y -depth $DEPTH -mag $MAG -i 17
python ../src/reg_inference.py -g 0 -m ../result/result_20230807_171616/model_best -x $X -y $Y -depth $DEPTH -mag $MAG --inputwidth 5 --inputdim 10
python ../src/hybrid_inference.py --gpu1 0 --gpu2 1 --regmodel ../result/result_20230807_171616/model_best --clsmodel ../result/result_20230807_183130/model_best --reginputwidth 5 --reginputdim 10 --clsinputwidth 17 --clsinputdim 1 -x $X -y $Y -depth $DEPTH -mag $MAG