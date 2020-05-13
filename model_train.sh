#!/usr/bin/env sh

# =====================================================
# Description: model_train.sh
#
# =====================================================
#
# Created by YongBai on 2020/2/13 11:33 PM. whole_train

echo "python model_train.py -t split_train">run_1.sh

qsub -cwd -P P18Z10200N0124 -q st_gpu1.q -l vf=1g,p=1 run_1.sh
#qsub -cwd -P P18Z10200N0124 -q st.q -l vf=10g,p=1 run_1.sh