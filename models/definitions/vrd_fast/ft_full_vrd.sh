cd /home/econser/usr/py-faster-rcnn
time ./tools/train_net.py --gpu 0 \
     --weights /home/econser/usr/py-faster-rcnn/output/default/train/vrd_fast_iter_0.caffemodel \
     --imdb vrd_fast_train \
     --cfg /home/econser/research/irsg_psu_pdx/models/model_definitions/vrd_fast/config.yml \
     --solver /home/econser/research/irsg_psu_pdx/models/model_definitions/vrd_fast/solver_full.prototxt \
     --iter 100000
