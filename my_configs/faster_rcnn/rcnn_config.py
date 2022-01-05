_base_ = './configs/faster_rcnn/faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco.py'

# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1)
    ),
    train_cfg=dict(
        rpn_proposal=dict(
            nms=dict(iou_threshold=0.9)
        )    
    ),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.5,
            nms=dict(iou_threshold=0.1),
            max_per_img=5
        )
    )
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
        img_scale=[(256, 256)],
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
        img_scale=(256, 256),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
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
            filter_empty_gt=True,
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
load_from = './weight/faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco_20210524_124528-26c63de6.pth'

# epoch
runner = dict(type='EpochBasedRunner', max_epochs=50)

# optimizer
optimizer = dict(lr=1e-4, weight_decay=5e-4)

# log config
log_config = dict(
    interval=250,
    hooks=[
        dict(type='TextLoggerHook')
    ]
)