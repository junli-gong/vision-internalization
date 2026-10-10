for u in https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip http://images.cocodataset.org/train2014/COCO_train2014_000000581857.jpg; do
curl -s -m 30 -r 0-52428799 -o /dev/null -w "$(hostname -s) $u http=%{http_code} size=%{size_download} speed=%{speed_download}B/s\n" "$u"; done
