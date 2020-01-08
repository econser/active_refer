cd /home/econser/usr/py-faster-rcnn
time ./tools/train_net.py --gpu 0 \
     --weights data/imagenet_models/VGG16.v2.caffemodel \
     --imdb vrd_fast_train \
     --cfg /home/econser/research/active_refer/models/definitions/vrd_fast/config.yml \
     --solver /home/econser/research/active_refer/models/definitions/vrd_fast/solver_init.prototxt \
     --iter 0
