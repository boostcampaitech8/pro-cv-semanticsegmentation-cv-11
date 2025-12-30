python /data/ephemeral/home/jsw_pro-cv-semanticsegmentation-cv-11/visual/visual_with_GT.py \
  --csv /data/ephemeral/home/pro-cv-semanticsegmentation-cv-11/for_vis/HRNet_W18/hrnet_w18-bce03_dice07-for_vis.csv \
  --image_root /data/ephemeral/home/dataset/fold0/val/DCM/ \
  --label_root /data/ephemeral/home/dataset/fold0/val/outputs_json/ \
  --ids ID004 ID009 ID014 ID313 ID338 ID543\
  --save_root /data/ephemeral/home/jsw_pro-cv-semanticsegmentation-cv-11/visual/vis_results \
  --only-class Pisiform Triquetrum