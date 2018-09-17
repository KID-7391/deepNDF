#!/usr/bin/env sh
set -e

sudo sh ./train_pi_first.sh
sudo sh ./train_theta.sh
for i in `seq 4`
do
	sudo sh ./train_theta.sh
	sudo sh ./train_pi.sh 
done
