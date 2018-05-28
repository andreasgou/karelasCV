#!/usr/bin/env bash

# INFORMATION
# -----------
#
# Put your positive images in the ./positive_images folder and create a list of them:
#
# find ./positive_images -iname "*.jpg" > positives.txt
#
# Put the negative images in the ./negative_images folder and create a list of them:
#
# find ./negative_images -iname "*.jpg" > negatives.txt




# SAMPLE SCRIPT
# -------------
VEC=$1  # vec output file name
INF=$2  # image annotations file name
BCK=$3  # backgrounds file descriptor
W=$4    # output img width
H=$5    # output img height
NUM=$6  # number of samples

if [ "$INF" = "view" ]; then
    opencv_createsamples -vec $VEC -show -w $W -h $H
else
    opencv_createsamples -vec $VEC -info $INF -bg $BCK -num $NUM -maxidev 10 -show -w $W -h $H
fi

# Annotation tool
opencv_annotation --annotations=$INF --images=$IMG



# Training tool
opencv_traincascade \
    -data data/thermometer/cascades \
    -vec data/thermometer/thermo.vec \
    -bg data/thermometer/thermo.bg.txt \
    -minHitRate 0.999 \
    -maxFalseAlarm 0.5 \
    -numPos 6 \
    -numNeg 1 \
    -numStages 10 \
    -precalcValBufSize 1024 \
    -precalcIdxBufSize 1024 \
    -featureType HAAR \
    -w 28 -h 34 \
    -mode ALL

# Notes on opencv_traincascade
# ----------------------------
# Lets make it more clear.
# When setting up a cascade classifier training using your command
# opencv_traincascade -numPos 7500 -numNeg 3000 -featureType LBP -mode ALL -numStages 12 -w 48 -h 48
# most of the deciding parameters are selected automatically.
#
# For example
#
# -minHitRate is set to 0.995 by default and -maxFalseAlarmRate is set to 0.5 by default.
# This means that for your current model, you allow 5 out of 1000 positive samples to get wrongly classified during
# the training process, whereas each stage needs to reach a individual false acceptance rate (good classification of
# negatives) that is a bit better than random guessing, symbolized by 0.5 value.
#
# So try one of the following things
#
# - Change -minHitRate to 0.998 for example and make it harder to reach your goals.
#   However this is not the best approach, since you force your model to overfit to the training data.
# - Change -maxFalseAlarmRate to for example 0.7, which forces individual stages to be more complex.
#   This is what I suggest if you dont want to add data.
#
# But like being suggested, best thing is to add more discriminative training data to the process.
