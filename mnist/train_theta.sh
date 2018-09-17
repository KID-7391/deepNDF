#!/usr/bin/env sh
set -e

/home/wen/DeepNDF/deepndf/build/tools/caffe train --solver=/home/wen/DeepNDF/mnist/df_solver_theta.prototxt -weights=/home/wen/DeepNDF/mnist/snapshot_pi/df_solver_pi_iter_3000.caffemodel --gpu 0 2>&1| tee /home/wen/DeepNDF/mnist/Log/caffe.log
