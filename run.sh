#!/bin/bash
for i in {100,200,400,600,800}
do
	for j in {1,2,4,8,16}
	do
		echo $i $i $i $j
		echo "************ cublas ***********"
		./cublas $i $i $i $j
		echo "************* mine ************"
		./9-5 $i $i $i $j
		echo "*****************************************"

		python verify.py gpu.out cublas.out
		echo "*****************************************"
		echo ""
		echo ""
	done
done
