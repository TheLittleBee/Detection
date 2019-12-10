Contents

[Models](#models) <br>
[Network](#network)

---

## Models
- [x] YOLOv3
- [x] YOLOv3_MobileNetv2
- [x] SSD300,512
- [x] SSDLite
- [x] RetinaNet_ResNet
- [x] RetinaNet_MobileNetv2
- [x] Faster_RCNN_ResNet
- [x] Cascade_RCNN

## Network

### Backbone
Support load pretrained model from pytorch model zoo.
Only for ResNet, ResNeXt, Inceptionv3, ShuffleNetv2.
The rest backbone use our own model zoo.

- [x] Darknet53
- [x] Darknet19
- [x] tinyYOLOv3
- [x] ResNet18,34,50,101,152
- [x] ResNeXt50,101
- [x] Inceptionv3
- [x] MobileNetv1,v2
- [x] ShuffleNetv2
- [x] VGG16, w/ BN

### Head
- [x] YOLOv3
- [x] SSD
- [x] SSDLite
- [x] RetinaHead
- [x] RCNNHead
- [x] RPN
- [ ] MaskHead

### Neck
- [x] FPN

### RoI
- [x] RoIpool
- [x] RoIalign

### Plugins
- [x] CBAM
- [x] Dual Attention
- [x] Non-local
- [x] SE
- [x] ASPP