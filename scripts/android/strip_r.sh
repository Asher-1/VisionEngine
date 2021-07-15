#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Please, provide the strip sh script"
    exit 1
fi

sed -i 's/\r//' $1
