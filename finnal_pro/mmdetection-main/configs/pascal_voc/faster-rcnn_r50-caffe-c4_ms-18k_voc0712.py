_base_ = [
    '../_base_/models/faster-rcnn_r50-caffe-c4.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/datasets/voc07.py',
    '../_base_/default_runtime.py'
]

# runner = dict(type='EpochBasedRunner', max_epochs=7)
model = dict(roi_head=dict(bbox_head=dict(num_classes=20)))



log_processor = dict(by_epoch=True)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # Wandb logger
        dict(type='WandbLoggerHook',
             init_kwargs=dict(
                 project='pure_VOC_training',
                 name='faster_rcnn',  # 实验名称
             )),
        dict(type='TensorboardLoggerHook')
    ])

_base_.visualizer.vis_backends = [
    dict(type='LocalVisBackend'), #
    dict(type='TensorboardVisBackend'),
    dict(type='WandbVisBackend'),]

