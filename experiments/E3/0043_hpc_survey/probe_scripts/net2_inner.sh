for u in http://images.cocodataset.org/annotations/annotations_trainval2014.zip https://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip https://raw.githubusercontent.com/cleinc/bts/5e3406b35d1497b2e55d2dd600524d1f4efacaed/train_test_inputs/nyudepthv2_train_files_with_gt.txt https://drive.usercontent.google.com/ https://web.archive.org/web/ https://bvisionweb1.cs.unc.edu/licheng/referit/data/; do
  r=$(curl -sI -o /dev/null -m 20 -w "%{http_code} %{time_total}s" "$u" 2>&1); echo "$(hostname -s) HEAD $u -> $r"
done
curl -s -m 40 -r 0-52428799 -o /dev/null -w "$(hostname -s) coco 50MB range: http=%{http_code} size=%{size_download} speed=%{speed_download}B/s\n" http://images.cocodataset.org/zips/train2014.zip
