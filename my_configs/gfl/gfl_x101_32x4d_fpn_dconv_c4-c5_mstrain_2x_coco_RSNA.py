_base_ = './gfl_r50_fpn_mstrain_2x_coco.py'
model = dict(
    type='GFL',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_32x4d')),
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

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)

load_from = "checkpoints/gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco_20200630_102002-14a2bf25.pth"
