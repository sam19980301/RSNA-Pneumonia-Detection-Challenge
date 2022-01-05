_base_ = './configs/swin/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py'


model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)
    ),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.5,
            nms=dict(iou_threshold=0.2),
            max_per_img=10
        )
    )
)

# dataset
classes = ('1',)
dataset_type = 'COCODataset'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        filter_empty_gt=True,
        img_prefix='../data/train/',
        classes=classes,
        ann_file='../data/train.json'),
    val=dict(
        filter_empty_gt=False,
        img_prefix='../data/valid/',
        classes=classes,
        ann_file='../data/valid.json'),
    test=dict(
        img_prefix='../data/test/',
        classes=classes,
        ann_file='../data/test.json')
    )


# epoch
runner = dict(type='EpochBasedRunner', max_epochs=50)

# optimizer
optimizer = dict(
    #lr=1e-4,
    weight_decay=1e-3
)


# log config
log_config = dict(
    interval=500,
    hooks=[
        dict(type='TextLoggerHook')
    ]
)

# evaluation
evaluation = dict(metric=['bbox'])