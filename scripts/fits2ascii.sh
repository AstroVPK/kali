#!/bin/bash
read -e -p "Path to Kepler data: " -i "/home/exarkun/Documents/Research/Kepler/" keplerPath
cd "${keplerPath}"
for object in ./kplr*/; do
    echo $object
    cd "${object}"llc
    cp *.fits .. 
    cd .. 
    for f in *fits; do
        fb=${f##*/}
        fb=${fb%.fits}
        fb=$fb.dat
        topcat -stilts tcopy ${f##*/} ${fb} ofmt=ascii
        done 
    rm *fits
    cd ..
    done