# 1st Place Solution - Finetuning and Ensemble Multiple Mammogram Pretrained SOTA Models


**First of all, thanks for organizing this wonderful competition!**

I'm honored to share our 1st place solution for the SPR Screening Mammography Recall competition. The goal was to predict whether a screening mammogram leads to a recall (BI-RADS 0, 3–5) or not (BI-RADS 1–2).
Working with such an interesting and clinically relevant topic related to breast cancer screening was very rewarding.
Here is our solution:

### Overview of our pipeline.
![Pipeline](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F7571336%2F04a2c764c704b8207f67c6994c0cdd0c%2Fpipeline.png?generation=1746218869925533&alt=media)

Our final solution combines multiple pretrained mammography-specific backbones, finetuned on SPR dataset, and then ensembles to boost performance.

Specifically, each individual undergoes a screening mammogram, typically with left and right breasts, and each side has multiple views (e.g., CC and MLO).
1.  **Multi-model inference:** 
We apply an ensemble of multiple finetuned models: 
(e.g., EfficientNet-B2/B5, ConvNeXt-small, and ResNet18 to each view independently, outputting probabilities indicating the likelihood of recall.

2.  **Breast-level score aggregation:**
For each breast (left/right), we compute the average of the predicted probabilities from all views to get a breast-level score.

3.  **Patient-level score aggregation:**
Finally, we take the maximum of the two breast scores as the patient-level score.
This design decision empirically improved AUC by ~0.01 (e.g., from 0.783 → 0.793 on public LB).
The final patient score is used to classify whether the patient should be recalled or not.
---
### Data Preprocessing
1. Converted DICOM to PNG with both raw and processed (cropped + resized) formats.
2. Used high-resolution images (1536x768, 2048x1024) to retain fine mammographic details.
---
### Model training.
1. Rather than training from scratch, we tried to finetune pretrained backbones from SOTA mammography models on the SPR dataset.
   - EfficientNet-B2/B5 from [Mammo-Clip](https://github.com/batmanlab/Mammo-CLIP) - pretrained on a large mammogram dataset with radiology reports via clip.
   - ConvNeXt-Small ([RSNA2023-1st-Team-Solution](https://github.com/dangnh0611/kaggle_rsna_breast_cancer)) – which was developed on the RSNA2023 challenge dataset for the breast cancer classification task.
   - ResNet18 from [MIRAI](https://github.com/yala/Mirai) – a robust breast cancer risk prediction.
2. Each was fine-tuned using a 4-fold cross-validation split (patient-level)
3. We also tried to use different image sizes (1536x768, 2048x1024) to train the models.
4. We finally selected four strong models according to the validation AUC (For each fold, we select the top one model).
   - **Fold 1 ConvNeXt-Small - trained on 2048x1024 image size**
   - **Fold 2 EfficientNet-B5 - trained on 1536x768 image size**
   - **Fold 3 EfficientNet-B2 - trained on 1536x768 image size**
   - **Fold 4 ConvNeXt-Small - trained on 2048x1024 image size**
---
### Experience and tricks learned from the challenge
#### ✅ What works
- **Ensemble model**. Make sure each fold-based model is involved in the ensemble model, _even if no models in the same fold are good (not very sure)._
- **Large size of the image sometimes works better.**
- **Advancement of the backbone model.** For example, EfficientNet-B2/B5 and ConvNext-Small are better than ResNet18. May be due to the more parameters and optimized architectural design.
- I changed the averaging probs to maximum prob when calculating the patient-level scores from breast-level scores, AUC ups from 0.783 to 0.793!

#### ❌ What doesn't work
- Aux-task learning is not working well for this task.
- More external training datasets are not working well. Maybe I failed to set the label correctly.

### More details of the result-logs and code can be found in our [GitHub](https://github.com/xinwangxinwang/Kaggle_SPR)

---
## 🙏 Acknowledgements
Thanks to:
- [Kaggle SPR Screening Mammography Recall](https://www.kaggle.com/c/spr-screening-mammography-recall/overview)
- [RSNA Screening Mammography Breast Cancer Detection](https://www.kaggle.com/competitions/rsna-breast-cancer-detection)
- [Github: Mammo-Clip](https://github.com/batmanlab/Mammo-CLIP)
- [Github: kaggle_rsna_breast_cancer](https://github.com/dangnh0611/kaggle_rsna_breast_cancer)
- [Github: Mirai](https://github.com/yala/Mirai)


Feel free to ask any questions! Happy to discuss design choices or implementation details. Good luck to future mammography challenges!


