#!/usr/bin/env sh

# =====================================================
# Description: loaddata.sh
#
# =====================================================
#
# Created by YongBai on 2020/1/20 2:37 PM.

echo "python loaddata.py -t train_data">run_1.sh

qsub -cwd -P P18Z10200N0124 -q st_gpu1.q -l vf=1g,p=1 run_1.sh
#qsub -cwd -P P18Z10200N0124 -q st.q -l vf=10g,p=1 run_1.sh