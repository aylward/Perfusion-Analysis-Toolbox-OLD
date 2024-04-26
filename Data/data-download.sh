#! /bin/sh

# Please install girder-client via
#   pip install girder-client

# Data is downloaded from
#   https://data.kitware.com/#collection/661410cf2357cf6b55ca8bda

girder-client --api-url https://data.kitware.com/api/v1 download 661410eb2357cf6b55ca8bdb .
