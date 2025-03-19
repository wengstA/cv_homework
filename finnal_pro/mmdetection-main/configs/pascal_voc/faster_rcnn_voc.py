_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn_voc.py',
    '../_base_/datasets/voc0712.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

runner = dict(type='EpochBasedRunner', max_epochs=7)