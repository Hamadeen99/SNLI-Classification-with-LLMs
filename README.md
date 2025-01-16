# SNLI Classification with LLMs

## Overview
This project demonstrates the use of large language models (LLMs) for **Natural Language Inference (NLI)** tasks, leveraging the Stanford Natural Language Inference (SNLI) dataset. The project fine-tunes a pre-trained LLM using **LoRA (Low-Rank Adaptation)** for efficient training and classifies relationships between sentence pairs into one of three categories:
1. **Entailment**
2. **Neutral**
3. **Contradiction**

The model selected for this task is `microsoft/Phi-3-mini-4k-instruct`, but other models like Llama-2 and Llama-3 can also be used.

---

## Features
- **Efficient Fine-Tuning**: Implements LoRA for lightweight and resource-efficient updates to pre-trained LLMs.
- **Prompt-Based Training**: Converts the dataset into prompts for natural language interaction.
- **Hugging Face Integration**: Uses Hugging Face's `transformers` library and datasets for seamless pipeline integration.
- **Quantization Support**: Supports 4-bit or 8-bit quantization for reduced memory usage.

---

## Dependencies
- Python 3.7+
  
   ```Requirments
   pip install torch transformers peft trl pandas tqdm
   ```


---

## Usage

### Steps:
1. **Prepare Environment**:
   - Replace the Hugging Face token in the script with your own token.

2. **Run the Script**:
   ```bash
   python SNLIClassification.py
   ```

3. **Adjust Configurations**:
   Modify the following parameters in the script for custom runs:
   - `model_name`: Model to be fine-tuned (e.g., Phi-3, Llama-3).
   - `batch_size`: Batch size for training.
   - `seq_length`: Sequence length for inputs.
   - `num_train_epochs`: Number of training epochs.

---

## Example
### Input:
**Premise**: "A person on a horse jumps over a broken-down airplane."  
**Hypothesis**: "A person is training his horse for a competition."

### Output:
**Label**: "Neutral"

---

## Results
The fine-tuned model achieves the following:
- **High Accuracy**: With a single epoch of training, accuracy reaches **98.38%** using the Phi-3-mini-4k-instruct model.

---

## Repository Structure
```
SNLIClassification/
├── SNLIClassification.py       # Main script for fine-tuning and evaluation        
├── data/                       # Folder for SNLI dataset
├── checkpoints/                # Folder for saving model checkpoints
├── README.md                   # Project documentation
```

---

## Key Components

1. **Dataset Handling**:
   - Converts the SNLI dataset into prompts for training and evaluation.
   - Example training prompt:
     ```
     ### Human: Classify the relationship between the following two sentences as one of the following: entailment, neutral, contradiction.
     Premise: A person on a horse jumps over a broken-down airplane.
     Hypothesis: A person is training his horse for a competition.
     ### Assistant: Neutral
     ```

2. **LoRA Configuration**:
   - Applies lightweight updates to linear layers of the model for efficient fine-tuning.

3. **Quantization**:
   - Reduces memory usage via 4-bit or 8-bit quantization.

---

## Acknowledgments
This project utilizes the Hugging Face Transformers library and the SNLI dataset. LoRA fine-tuning techniques are applied for the efficient adaptation of large language models.

---
