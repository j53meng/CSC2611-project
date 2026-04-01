#!/bin/bash

# This example downloads the English sgns embeddings 

mkdir embeddings
cd embeddings
curl -o eng-all_sgns.zip http://snap.stanford.edu/historical_embeddings/eng-all_sgns.zip
unzip eng-all_sgns.zip
mv sgns eng-all_sgns
cd ..
