# Features Extraction
This model can be used for extracting visual features useful for Relational Content-Based Image Retrieval applications. Along with the original RN formulation, we employed a tiny modification, called 2S-RN (Two-stage RN):

![twolayerrn](https://user-images.githubusercontent.com/25117311/44774024-5c6bf300-ab72-11e8-9258-52aaa5805f64.png)

In order to extract visual features, the simplest way is to run the following:
```
./extract_features.sh path/to/CLEVR_v1.0
```
Features are stored in pickle format under ```features``` folder.

Otherwise, you have first to train the model using our two-stage RN architecture for IR (follow steps in README for how to train the model).
If you have not enough computing resources, you can use our **pretrained model** for IR (```ir_fp_epoch_312.pth```).

You can extract features using our pretrained model using the following command:
```
python3 extract.py --clevr-dir ../../../CLEVR_v1.0/ --model 'ir-fp' --checkpoint pretrained_models/ir_fp_epoch_312.pth
```

By default, features are extracted after the 2-nd layer of g. You can try extracting features from a different layer by acting on the ```--extr-layer-idx``` option.
