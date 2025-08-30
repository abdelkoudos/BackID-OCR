# Egyptian National ID Back OCR (Arabic Digits) with 95.5% line accuracy

## Project Overview
This project implements an end-to-end pipeline for detecting and recognizing the 14-digit number printed on the back of Egyptian national ID cards. The pipeline focuses on robust preprocessing, segmentation, and classification to achieve high accuracy in character and line recognition.

## Pipeline

### Preprocessing
- Remove duplicates and wrongly labeled images  
- Filter out invalid ID text and pictures  

### Segmentation
- Resize and convert to grayscale  
- Adaptive thresholding with Gaussian filter  
- Morphological closing and dilation  
- Contour detection to isolate each digit  

### Character Classification
- Resized, normalized, and augmented digit crops  
- CNN classifier trained on segmented digits  
- Iterative data cleaning using model inference and retraining  

## Dataset
- **Initial dataset**: 2,134 lines  
- **After cleaning**: 1,620 lines  

**Data split:**  
- Training: 1,134 lines → 15,834 digit crops  
  - Train: 9,500  
  - Validation: 3,167  
  - Test: 3,167  
- Test set: 486 lines  

## Results

### Character-level performance
- **Training accuracy:** 99.90%  
- **Validation accuracy:** 100%  
- **Test accuracy:** 99.96%  

### End-to-end line recognition
- **Training:** 99.18% char | 95.93% line  
- **Test:** 98.98% char | 94.63% line  -> 95.5% line
> *Note: The test set had wrong annotated samples. After testing on a cleaned test set, accuracy increased to 95.5%.*  

## Error Analysis
- Wrong segmentation
- Wrong annotations  
- Classification confusions (commonly 2 ↔ 3, 7 ↔ 8)  
- Unclear or blurry samples  

## Key Takeaways
- Iterative cleaning and retraining significantly improved dataset quality and performance.  
- Errors mainly arise from segmentation mistakes, ambiguous digits, or annotation issues.  
- Overall system achieves **>94% line-level accuracy** on test data.  

---

⚠️ **Note:** The dataset used in this project is confidential and therefore not included in this repository.
