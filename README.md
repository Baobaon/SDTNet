# SDTNet
-------------------------------------------------------Paper-------------------------------------------------------
A Triple-Branch Model with Style-Unified Contrastive Learning and Adaptive Dice Focal Loss for change detection
under review by TGRS

Abstract
Deep learning has significantly reshaped the landscape of remote sensing change detection (RSCD). However, detecting subtle changes under complex backgrounds remains a formidable challenge. To address this issue, Style-Unified Temporal Difference Contrastive Learning Strategy (STDCL) and Adaptive Dice Focal Loss (ADFLoss) are proposed to enhance the model's capability in extracting discriminative representations of subtle changes within intricate scenarios. Leveraging the strengths of CNN and Transformer architectures complimentarily, we design a Dual-Branch Temporal Difference Transformer (DTDFormer) to filter background noise from Siamese encoder outputs and amplify changed features, thereby constructing a three-branch model (SDTNet) based on STDCL and DTDFormer. Specifically,  an auxiliary encoder based on a Domain Consistency Module (DCM) is introduced to generate multi-scale domain-aligned difference features, while the hybrid CNN-Transformer Siamese encoder produces multi-scale unaligned difference features. Although domain adapter approaches are conceptually straightforward, existing models often incur substantial computational overhead, limiting their practicality. To address this, we propose a Difference-Enhanced Encoder (DEE) that efficiently extracts domain-aligned difference features. This is complemented by a contrastive learning strategy that projects original difference features into the representation space of domain-aligned positive samples. Finally, the ADFLoss adaptively balances class weights and enhances the model's sensitivity to subtle changes in complex backgrounds. Experimental results demonstrate that our SDTNet achieves state-of-the-art (SOTA) performance across five large-scale change detection datasets.
[f6.pdf](https://github.com/user-attachments/files/21344420/f6.pdf)

--------------------------------------------------run Environment--------------------------------------------------
Please see requirement.txt


------------------------------------------------------pretrain------------------------------------------------------
Please see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

Our conda environment seems to conflict with that of CycleGAM. You should configure the pre-training environment according to their environment settings.

------------------------------------------------------new------------------------------------------------------
2025-07-21 We have released the preliminary code, and the subsequent complete code will be released soon after passing the TGRS review.



