
# ---------------- Demo ----------------
# pre-train with cifar10
python pretrain_moco_unet_cifar.py \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  ./data
# ---------------- Demo ----------------


# pre-train with imagenet format data
python pretrain_moco_unet.py \
  --lr 0.03 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  ./data

# train
python train.py --epochs 40 -batch-size 128