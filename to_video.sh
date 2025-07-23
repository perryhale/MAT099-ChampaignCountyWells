#!/bin/bash
ffmpeg -framerate $2 -i $1%d.png -c:v libx264 -preset veryslow -crf 0 -pix_fmt yuv444p out.mp4
