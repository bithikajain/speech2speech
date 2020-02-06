#!/usr/bin/env bash

export RUN_DIR=$1

cd $RUN_DIR

for f in $(/bin/ls *.sh) ; do
    bash $f ;
done
