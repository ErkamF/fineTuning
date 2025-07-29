# 🤖 BERT Fine-Tuning for Text Classification

This project demonstrates how to fine-tune the `bert-base-uncased` model for binary sentiment classification using the IMDb dataset. The training pipeline is built using [Hugging Face Transformers](https://github.com/huggingface/transformers) and the [datasets](https://github.com/huggingface/datasets) library.

## 📌 Project Structure

- Fine-tunes `bert-base-uncased` on a small subset of IMDb
- Uses PyTorch backend with Hugging Face `Trainer`
- Saves and reloads the trained model
- Includes a basic text classification pipeline

## 🛠️ Requirements

Install the dependencies with:

```bash
pip install transformers datasets torch scikit-learn
