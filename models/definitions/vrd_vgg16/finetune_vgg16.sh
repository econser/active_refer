#cd /home/econser/usr/py-faster-rcnn/caffe-fast-rcnn
cd /home/econser/usr/caffe
time ./build/tools/caffe train -gpu 1 \
     -weights /home/econser/research/irsg_psu_pdx/models/model_definitions/vrd_vgg/VGG_ILSVRC_16_layers.caffemodel \
     -solver /home/econser/research/irsg_psu_pdx/models/model_definitions/vrd_vgg/solver_init.prototxt 1>&2 | tee vrd_vgg_train.log
