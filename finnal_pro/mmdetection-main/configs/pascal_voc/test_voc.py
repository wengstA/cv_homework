_base_ = [
    '../_base_/models/faster-rcnn_r50-caffe-c4.py',
    '../_base_/default_runtime.py'
    ]
dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'
classes = [ 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car','elephant',
                    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

backend_args = None

runner = dict(type='EpochBasedRunner', max_epochs=7,grad_clip=dict(max_norm=35, norm_type=2))
model = dict(roi_head=dict(bbox_head=dict(num_classes=20)))

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    # avoid bboxes being resized
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='VOC2007/ImageSets/Main/test_voc_coco.txt',
        # ann_file='VOC2007/Annotations/VOC_COCO_fmt.json',
        data_prefix=dict(sub_data_root='VOC2007/'),
        # img_prefix='VOC2007/JPEGImages/',
        metainfo=dict(classes=classes),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
# test_dataloader = val_dataloader
# test_evaluator = dict( type='VOCMetric',metric='mAP', eval_mode='11points')
test_cfg = dict(type='TestLoop')

test_evaluator = dict( type='VOCMetric',metric='mAP', eval_mode='11points')
# test_evaluator = val_evaluator
