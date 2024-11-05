
# SBP and DBP Prediction Using Llama3.2

This repository provides an implementation to fine-tune and deploy a Llama3.2 model for predicting systolic blood pressure (SBP) and diastolic blood pressure (DBP) values. It uses physiological input features (like CO, PTTs, and PRs) to predict SBP and DBP as continuous values, ideal for healthcare applications.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Model Training](#model-training)
5. [Evaluation](#evaluation)
6. [Deployment](#deployment)
7. [Acknowledgements](#acknowledgements)

---

### Project Overview

Predicting blood pressure accurately can aid in early disease detection and intervention. This project uses Llama3.2, an advanced language model, fine-tuned to predict SBP and DBP values. The model takes physiological features (e.g., Cardiac Output (CO), Pulse Transit Times (PTTs), and Pulse Rate (PRs)) as input to provide accurate blood pressure predictions.

### Installation

#### Prerequisites

- Python 3.8+
- PyTorch
- Hugging Face Transformers
- Other dependencies listed in `requirements.txt`

#### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/poku0857643/Llama3.2_BP.git
   cd Llama3.2_BP
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure MLflow tracking for model training and evaluation:
   ```bash
   mlflow ui
   ```

### Data Preparation

1. **Physiological Features**: Ensure your dataset contains physiological features like CO, PTTs, and PRs required for the prediction task.
2. **Formatting**: Format the dataset as follows:
   ```json
   {
       "input": "The physiological features are CO, PTTs, and PRs. CO: [values], PTTs: [values], PRs: [values].",
       "target": "The SBP value is 135.02, and the DBP value is 51.66."
   }
   ```
3. **Tokenizer Setup**: Tokenize the input text and the target values (SBP and DBP) as shown in `tokenize_function` in `scripts/tokenization.py`.

4. **Storage**: Separate SBP and DBP predictions if required. Save formatted data in a `.json` file for fine-tuning.

### Model Training

1. Configure training arguments in `config/training_args.json`, specifying batch size, learning rate, and other hyperparameters.

2. **Run the Training Script**:
   ```bash
   python scripts/train.py --config config/training_args.json
   ```
   - The model uses Mean Squared Error (MSE) loss for regression.
   - Training logs and metrics (e.g., MAE, RMSE) are tracked using MLflow.

3. **Prompt-based Fine-Tuning**:
   Fine-tune using prompt-based PEFT with QLoRA for efficient fine-tuning. Define `LoraConfig` in `scripts/lora_config.py` and run:
   ```bash
   python scripts/fine_tune_with_lora.py
   ```

### Performance Evaluation

1. After training, evaluate the model on a test set by running and 

2. Plot metrics such as Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to assess performance.

3. **Visualization**: Use TensorBoard or MLflow to view detailed metrics and analyze performance.

### Deployment

To deploy the model as an API on a cloud service:

1. **Build the API**:
   Use FastAPI or Django to create an endpoint for predictions. Example:
   ```python
   from fastapi import FastAPI
   from predict import predict_sbp_dbp  # Import prediction function

   app = FastAPI()

   @app.post("/predict")
   async def predict(input_data: dict):
       return predict_sbp_dbp(input_data)
   ```

2. **Deploy on Cloud**: # 
   - Use Docker to containerize the application.
   - 

3. **Monitoring**:
   implement logging and monitoring to track model performance and API latency.

### Acknowledgements

This project is built using the [Llama3.2 model](https://huggingface.co/) and the Hugging Face Transformers library. Special thanks to the community for open-source tools and resources.

