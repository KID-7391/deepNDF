#!/usr/bin/env sh
set -e

/home/wen/DeepNDF/deepndf/build/tools/caffe train --solver=/home/wen/DeepNDF/cifar10/df_solver_theta.prototxt.prototxt --weights=/home/wen/DeepNDF/cifar10/snapshot_pi/df_solver_pi_iter_2500.caffemodel --gpu 0 2>&1| tee /home/wen/DeepNDF/cifar10/Log/caffe.log
