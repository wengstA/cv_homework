_base_ = [
    '../_base_/models/faster-rcnn_r50-caffe-c4.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/datasets/voc_coco.py',
    '../_base_/default_runtime.py'
]

runner = dict(type='EpochBasedRunner', max_epochs=7,grad_clip=dict(max_norm=35, norm_type=2))
model = dict(roi_head=dict(bbox_head=dict(num_classes=20)))


lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[3, 5])  # Adjust the learning rate schedule as needed
optimizer = dict(type='SGD', lr=0.00025, momentum=0.9, weight_decay=0.0001,grad_clip=dict(max_norm=35, norm_type=2))
log_processor = dict(by_epoch=True)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # Wandb logger
        dict(type='WandbLoggerHook',
             init_kwargs=dict(
                 project='coco_VOC_training',
                 name='faster_rcnn',  # 实验名称
             )),
        dict(type='TensorboardLoggerHook')
    ])

_base_.visualizer.vis_backends = [
    dict(type='LocalVisBackend'), #
    dict(type='TensorboardVisBackend'),
    dict(type='WandbVisBackend',
        init_kwargs=dict(
         project='coco_VOC_training',
         name='faster_rcnn',  # 实验名称
         )),
    ]

