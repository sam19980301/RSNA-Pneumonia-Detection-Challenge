_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    bbox_head=dict(
        num_classes=1
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.2,
        nms=dict(type='nms', iou_threshold=0.1),
        max_per_img=100)
)

dataset_type = 'COCODataset'            
classes = ('1',)

img_norm_cfg = dict(
    mean=[124.450, 124.450, 124.450], std=[58.427, 58.427, 58.427], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(256, 256), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=16,
    train=dict(
        img_prefix='../train_images',
        classes=classes,
        ann_file='../RSNA_train_0.json',
        filter_empty_gt=False,
        pipeline=train_pipeline),
    test=dict(
        img_prefix='../test_images',
        classes=classes,
        ann_file='../RSNA_answer.json',
        filter_empty_gt=False,
        pipeline=test_pipeline),
    val=dict(
        img_prefix='../train_images',
        classes=classes,
        ann_file='../RSNA_val_0.json',
        filter_empty_gt=False,
        pipeline=test_pipeline),
)


# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001) # default_lr = 0.02 / (gpus*sample_per_gpu/16)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

load_from = "checkpoints/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth"