```
python3 infer.py --model test --images /mnt/store/rolan/dataset/cityscapes_rgbd_segment/leftImg8bit/train/*/*.png
```

```buildoutcfg
python3 train.py --logdir test --classes 8 --base False --epochs 1 \
--input /mnt/store/rolan/dataset/cityscapes_rgbd_segment/leftImg8bit/*a*/*/*.png \
--output /mnt/store/rolan/dataset/cityscapes_rgbd_segment/gtFine/*a*/*/*_gtFine_labelIds.png
```
