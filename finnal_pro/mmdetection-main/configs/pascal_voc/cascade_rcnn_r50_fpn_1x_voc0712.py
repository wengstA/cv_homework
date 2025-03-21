_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn_voc.py',
    '../_base_/datasets/voc07.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

runner = dict(type='EpochBasedRunner', max_epochs=7)
