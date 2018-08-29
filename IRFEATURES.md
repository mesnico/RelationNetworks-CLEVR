# Features Extraction
This model can be used for extracting visual features useful for Relational Content-Based Image Retrieval applications.
In order to extract visual features, you have first to train the model using our two-stage RN architecture for IR.

![twolayerrn](https://user-images.githubusercontent.com/25117311/44774024-5c6bf300-ab72-11e8-9258-52aaa5805f64.png)

Follow steps in README for how to train the model.
Otherwise, you can use our **pretrained model** for IR (```ir_fp_epoch_312.pth```).

You can extract features using our pretrained model using the following command:
```
python3 extract.py --clevr-dir ../../../CLEVR_v1.0/ --model 'ir-fp' --checkpoint pretrained_models/ir_fp_epoch_312.pth
```

By default, features are extracted after the 2-nd layer of g. You can try extracting features from a different layer by acting on the ```--extr-layer-idx``` option.
