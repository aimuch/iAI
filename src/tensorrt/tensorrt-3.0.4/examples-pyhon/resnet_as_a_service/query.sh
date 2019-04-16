#!/bin/bash

function usage()
{
    echo "Simple script to upload an image to Resnetaas and return a top5 classification"
    echo ""
    echo "./query.sh [-h | --help] PATH TO IMAGE"
    echo ""
}

case $1 in
    -h | --help)
        usage
        exit
        ;;
    "")
        echo "ERROR: Need a path to image"
        usage
        exit 1
        ;;
    *)
        curl -X POST -F file=@$1 http://localhost:5000/classify
        exit
        ;;
esac