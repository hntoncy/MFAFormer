MFAFormer: Multi-scale feature enhancement with axial Transformer for medical image segmentation
论文地址：https://doi.org/10.1016/j.asoc.2026.115792
Medical image segmentation is fundamental to biomedical analysis, supporting diagnosis, lesion localization, organ mapping, and treatment planning, yet challenges persist, such as boundary ambiguity, lesion shape/size variability, and imprecise target localization.The development of U-shaped architectures for segmentation is constrained by a key trade-off that CNNs are limited in their ability to capture long-range interactions, whereas Transformers achieve this at the expense of high computational complexity and a loss of fine-grained spatial information. To overcome these limitations, this paper introduces a novel multi-scale feature enhancement method with axial Transformer for medical image segmentation, termed MFAFormer, which effectively integrates the local detail with global contextual information. MFAFormer consists of three novel modules, i.e., the multi-level feature supplement module (MLFS), the cross-scale feature enhancement module (CFE), and the dual-directional decoder via axial Transformer (D2AFormer), which enhance the feature representation for better segmentation performance. Firstly, the MLFS module selectively supplements and fuses multi-scale features of encoders by an aggregated attention-gated mechanism. Secondly, the CFE module further integrates the multi-attention mechanism to enhance the feature transmission. Finally, the D2AFormer module adopts a cyclical complementary structure, which jointly employs dual-directional convolutions and axial Transformers to improve the comprehensive feature representation ability for local details and long-range contextual information. Comprehensive experiments on public datasets are conducted to evaluate the proposed MFAFormer. The experimental results reveal substantial improvements over state-of-the-art methods, validating its potential for accurate medical image segmentation.
<img width="656" height="332" alt="image" src="https://github.com/user-attachments/assets/c1792797-4d7f-4658-ba80-dddfd419290c" />
The overview of the MFAFormer
<img width="621" height="416" alt="image" src="https://github.com/user-attachments/assets/94305d48-4221-4eb9-bf6c-6a8ed51fc73a" />
The proposed MLFS module.
<img width="459" height="550" alt="image" src="https://github.com/user-attachments/assets/9fbc1b87-e4a3-40c0-befc-134c8f3eb7f5" />
The proposed CFE module.
<img width="595" height="397" alt="image" src="https://github.com/user-attachments/assets/d1b04fd7-1e2f-47fb-9e65-c4bdfad5fb1e" />
The proposed D2AFormer module.
Citation:
Dongxu Cheng, Jingwen Zhang, Qiwei Dong, Yan Yang, Yuhui Zheng,
MFAFormer: Multi-scale feature enhancement with axial transformer for medical image segmentation,
Applied Soft Computing,
Volume 202, Part A,
2026,
115792,
ISSN 1568-4946,
https://doi.org/10.1016/j.asoc.2026.115792
