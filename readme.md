AI-Driven Workflow for Automated Colorimetric Isothermal Amplification for Robust Pathogen Detection on Portable Devices 

Here we provide an example of the workflow proposed in 'AI-Driven Workflow for Automated Colorimetric Isothermal Amplification for Robust Pathogen Detection on Portable Devices', which contains four steps of Cropping & Pre-processing, Enhancement & Course Segementation, Fine Segmentation & Cleaning, and Color Detection. Please run the codes in the following order:

1. cardDetect_20250113_new.py: detect AprilTags and crop raw photos into Cropped and Splited images.
2. Segmentation.ipynb: run enhancement, course and fine segmentations on splited tubes.
3. colorDetect.ipynb: clean the fine-segmented tubes, check empty and invalid controls, and idenfity targets.

A demo dataset is provided, without involving any clinical samples. Detailed descriptions can be found found in the Images folder.

Citation: Xu, Ke et al., "Ai-Driven Workflow for Automated Colorimetric Isothermal Amplification for Robust Pathogen Detection on Portable Devices." Available at SSRN: https://ssrn.com/abstract=4975777
