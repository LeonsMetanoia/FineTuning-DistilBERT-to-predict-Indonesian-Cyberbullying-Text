# Cyberbullying Detection with DistilBERT (Indonesian Language)

<p align="center">
  <img src="https://github.com/user-attachments/assets/7929b26e-3ca8-437f-bbfc-37eab8954964" alt="DataDrivenDecisionsForSuccessSuccessGIF" />
</p>


## Dataset

<p align="center">
  <img src="https://github.com/user-attachments/assets/c10c9da7-c7bd-4394-8d7a-c6c4e49cdf1f" alt="Sopranos Tony Soprano GIF" />
</p>


The data used is a combination of several secondary Indonesian-language datasets:

- Cyberbullying dataset from **Kaggle** by Cita Tiara Hanni (2021).
  https://www.kaggle.com/datasets/cttrhnn/cyberbullying-bahasa-indonesia
- Cyberbullying dataset from **HuggingFace** by aditdwi123 (2024), originally in JSON format and converted.
  https://huggingface.co/datasets/aditdwi123/cyber-bullying-dataset/tree/main
- Instagram comment dataset from **GitHub** by rizalespe (2019).
  https://github.com/rizalespe/Dataset-Sentimen-Analisis-Bahasa-Indonesia/blob/master/dataset_komentar_instagram_cyberbullying.csv
  
All data were merged and adapted for fine-tuning the model.

## Model Fine-tuning

The model used is the **DistilBERT Indonesian** from [cahya/distilbert-base-indonesian](https://huggingface.co/cahya/distilbert-base-indonesian).

Fine-tuning was performed on the combined dataset to detect cyberbullying in Indonesian text.

## Hyperparameter Tuning

Hyperparameter optimization was done using **Optuna** to achieve the best model performance on this classification task.

## Model Inference Speed Testing

This test code aims to evaluate how fast the fine-tuned DistilBERT model can perform predictions (inference) on text data, both in batch mode and real-time (single text).

### Main Steps:

1. **Load the saved fine-tuned model and tokenizer.**  
2. **Load a sample dataset** containing text comments to test (e.g., 500 random texts).  
3. **Batch inference:**  
   - Tokenize all texts at once (batch)  
   - Measure the start and end time of model inference without gradient calculation (to speed up)  
   - Calculate total time, average time per text (total time divided by number of texts), and throughput (number of texts divided by total time).  
4. **Real-time single-text inference:**  
   - Tokenize and infer on a single text  
   - Measure latency (inference time) for one text.

### Key Metrics:

- **Total batch time:** the elapsed time between start and end of inference on the whole batch.  
- **Average time per text**
