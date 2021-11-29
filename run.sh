#!/bin/bash

rm -rf main *.qdrep *.sqlite
nvcc -std=c++14 main.cu -o main
nsys nvprof main
