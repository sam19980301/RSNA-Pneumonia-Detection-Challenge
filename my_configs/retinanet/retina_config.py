_base_ = './configs/retinanet/retinanet_x101_64x4d_fpn_mstrain_640-800_3x_coco.py'

# model settings
model = dict(
    bbox_head=dict(
        num_classes=1,
        loss_cls=dict(
            gamma=2.0,
            alpha=0.6,
        )
    ),
    #test_cfg=dict(
    #    score_thr=0.05,
    #    nms=dict(iou_threshold=0.1),
    #    max_per_img=10
    #)
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# In mstrain 3x config, img_scale=[(1333, 640), (1333, 800)],
# multiscale_mode='range'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        #img_scale=[(1333, 640), (1333, 800)],
        img_scale=[(512, 512)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Rotate', level=10,  max_rotate_angle=6), # new add
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        #img_scale=(1333, 800),
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# dataset
classes = ('1',)
dataset_type = 'CocoDataset'
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            filter_empty_gt=False,
            img_prefix='../data/train/',
            classes=classes,
            ann_file='../data/train.json',
            pipeline=train_pipeline)),
    val=dict(
        filter_empty_gt=False,
        img_prefix='../data/valid/',
        classes=classes,
        ann_file='../data/valid.json',
        pipeline=test_pipeline),
    test=dict(
        img_prefix='../data/test/',
        classes=classes,
        ann_file='../data/test.json',
        pipeline=test_pipeline)
    )

# pretrained weight
load_from = './weight/retinanet_x101_64x4d_fpn_mstrain_3x_coco_20210719_051838-022c2187.pth'

# epoch
runner = dict(type='EpochBasedRunner', max_epochs=30)

# optimizer
optimizer = dict(lr=1e-5, weight_decay=1e-3)

# log config
log_config = dict(
    interval=250,
    hooks=[
        dict(type='TextLoggerHook')
    ]
)