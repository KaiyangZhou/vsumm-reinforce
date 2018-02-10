#!/bin/bash

"""
This script decompose a video into frames
How to use: replace path_to_videos and path_to_frames with real paths
"""

for f in path_to_videos/*.avi
do
  echo "Processing $f file..."
  # take action on each file. $f store current file name\
  basename=$(ff=${f%.ext} ; echo ${ff##*/})
  name=$(echo $basename | cut -d'.' --complement -f2-)
  echo $f
 mkdir -p path_to_frames/"$name"
 ffmpeg -i "$f" -f image2 path_to_frames/"$name"/%06d.jpg
done


