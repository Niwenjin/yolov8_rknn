# Flag-detect v0.4.0

Date: 2024.09.26

md5sum `b88d7a7fbb4b98675d2840488c8e12a4`

**Changes**

-   增加了负样本进行微调

# Flag-detect v0.3.0

Date: 2024.09.19

md5sum `b14a417b61a5d15999679ae3b5acbc12`

**Changes**

-   筛除了数据集中部分徽标，只关注完整的旗帜，重新训练模型
-   增加训练轮次到 1000
-   解决了对烟花的误检

# Flag-detect v0.2.0

Date: 2024.09.06

md5sum `ea1377452dcfae20cf0080999e9678b5`

**Changes**

-   新增了 4 个旗帜类别
-   新增了 mAP50 验证

# Flag-detect v0.1.0

Date: 2024.09.02

md5sum `07b5c0de9f51fe2aadbd360f8c7a406f`

**Changes**

-   选择 yolov8n 模型，参数量 3.2M，rknn 推理时间 93 ms
-   基于 coco 预训练模型进行训练
