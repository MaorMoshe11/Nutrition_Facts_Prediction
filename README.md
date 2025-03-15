# Caloric Prediction from RGB Image

**Submitted by:** Maor Moshe, David Oriel, and Shachar Ashkenazi  
**Submitted to:** Yuval Benjamini and Lee Carlin

---

## Abstract

Accurate nutritional analysis is essential for advancing health and dietary awareness. In this work, we explore predicting the caloric value of a dish from its RGB image—traditionally a task requiring complex datasets like Nutrition5k that include RGB images, depth images, and videos. Our approach simplifies this by using only RGB images while incorporating ingredient lists as an auxiliary task to enhance model performance. We investigate various Vision-Language Models (VLMs) and architectures that jointly process image and text embeddings, demonstrating that multi-modal learning can significantly improve the accuracy of caloric prediction. This work underscores the potential for democratizing nutrition information with a scalable solution for real-world applications.

---

## Introduction

Accurately estimating the nutritional content of meals is integral to maintaining overall health, yet even professional nutritionists struggle with assessing caloric and macronutrient values based solely on images. Variations in portion sizes, cooking methods, and hidden ingredients render purely visual estimations error-prone and labor-intensive. Recent advances in multimodal machine learning offer promising avenues to automate meal nutrition estimation. Inspired by Google’s 2021 Nutrition5k paper—which demonstrated superior nutritional prediction using a rich multi-modal dataset—this work focuses on predicting caloric values from RGB images alone, aiming to retain high accuracy while simplifying the data requirements.

---

## Data

The dataset is derived from the Nutrition5k collection and consists of RGB images (256×256 pixels). Originally containing 5,000 examples, our cleaning process revealed several unusable images and an intentionally hidden test set. Consequently, the final dataset comprises 2,600 training examples and 200 test examples, including information on meal caloric values, ingredients, and other pertinent details. The dishes vary widely in calorie count and ingredient complexity, reflecting real-world conditions but also presenting challenges such as occluded ingredients and diverse portion sizes.

---

## Methods

### Nutrition5k Model Overview

The original Nutrition5k model leverages RGB frames and depth images from rotating video sequences to predict food mass, calories, and macronutrient composition via a multi-task learning framework. Using an InceptionV2 encoder followed by fully connected layers and optimized with RMSProp, the model splits final layers for task-specific outputs while sharing a common feature representation.

### Proposed Methodologies

We explore several models to improve on the original approach:

- **Model 1 – Updated Original Architecture:**  
  Implements the Nutrition5k architecture with InceptionV3 replacing InceptionV2 for improved feature extraction.

- **Model 2 – Enhanced Complexity with ResNet and Attention:**  
  Integrates a 50-layer ResNet with an attention layer to better emphasize critical image features.

- **Model 3 – Augmented Input with Object Detection and Embeddings:**  
  Uses EfficientNet-B3 to detect and one-hot encode ingredients. This output is fused with features extracted by ResNet18 to predict caloric content.

- **Image-Text Fusion (Pipeline Model):**  
  A four-stage process:
  1. **Image:** Feature extraction via ResNet18.
  2. **Ingredient List:** Conversion to indices, embedding, and averaging.
  3. **Fusion:** Concatenation of image and text features.
  4. **Regression:** Caloric prediction using a small multilayer perceptron (MLP).

---

## Results

Our experiments compare models using metrics such as mean absolute error (MAE) and mean absolute percentage error (MAPE). The pipeline model (image-text fusion) significantly outperforms the pure image-based Inception model in MAPE (49% vs. 117%) and shows competitive performance in MAE. Although our pipeline model’s MAPE (48.69%) does not match the Nutrition5k benchmark (26.1%), it achieves better MAE performance, attributed to differences in dataset size, preprocessing, and computational complexity.

---

## Summary

This study explores the prediction of caloric values from RGB images using deep learning and multimodal approaches. By incorporating ingredient data into a vision-language pipeline, our models enhance caloric estimation accuracy compared to purely image-based methods. Despite not matching the full complexity of Nutrition5k, our work demonstrates the viability of a simplified, scalable approach to automated nutritional analysis, paving the way for improved real-time food logging and personal nutrition management.

---

## Repository Structure

### Jupyter Notebooks
- **Data prep.ipynb:** Preprocessing and cleaning of the dataset.
- **efficientnet_b3.ipynb:** Implementation and training of the EfficientNet-B3 model.
- **resenet_models_eval.ipynb:** Evaluation of ResNet-based models.
- **Evaluate_all_models.ipynb:** Comparative analysis across all models.
- **ResNet with attention and inceptionv3.ipynb:** Implementation of ResNet with attention mechanism and InceptionV3 model.

### Data Files
- **cafe1.csv**
- **cafe2.csv**

---

## Reference

Thames, Q., Karpur, A., Norris, W., Weyand, T., Xia, F., Sim, J., & Panait, L. (2021). [Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food](https://arxiv.org/pdf/2103.03375).


