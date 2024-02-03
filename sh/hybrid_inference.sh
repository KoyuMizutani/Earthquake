# python ../src/hybrid_inference.py --gpu1 0 --gpu2 1 --regmodel ../result/result_20230807_171616/model_best --clsmodel ../result/result_20230807_183130/model_best --reginputwidth 5 --reginputdim 10 --clsinputwidth 17 --clsinputdim 1 -x 38 -y 35 -depth 13.08 -mag 6.8
# for abnormal 59,0,386.940000,5.500000
# python ../src/hybrid_inference.py --gpu1 1 --gpu2 2 --regmodel ../result/result_20230807_171616/model_best --clsmodel ../result/result_20230807_183130/model_best --reginputwidth 5 --reginputdim 10 --clsinputwidth 17 --clsinputdim 1 -x 59 -y 0 -depth 386.94 -mag 5.5
python ../src/hybrid_inference.py --gpu1 0 --gpu2 1 --regmodel ../result/result_20230807_171616/model_best --clsmodel ../result/result_20230807_183130/model_best --reginputwidth 5 --reginputdim 10 --clsinputwidth 17 --clsinputdim 1 -x 37 -y 37 -depth 8.3 -mag 6.7