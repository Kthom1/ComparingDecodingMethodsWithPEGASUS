# Comparing Decoding Methods with PEGASUS

This Jupyter notebook provides a comprehensive framework for comparing different decoding methods using the PEGASUS model for text summarization. It specifically focuses on evaluating various decoding strategies on the CNN/DailyMail dataset.

## Features

- Implements multiple decoding methods:
  - Beam Search (with configurable beam sizes: 8 and 16)
  - Top-K Sampling (with K values: 40 and 640)
  - Top-K Sampling with Temperature (temperature = 0.7)
  - Nucleus Sampling (Top-P, P = 0.95)
- Compares two PEGASUS model variants:
  - PEGASUS-large (base model)
  - PEGASUS-CNN/DailyMail (fine-tuned model)
- Evaluates generated summaries using:
  - ROUGE metrics
  - Lexical diversity metrics (MTLD and HDD)

## Setup Requirements

1. **Hardware Requirements**:
   - GPU support (the notebook checks for CUDA availability)
   - Google Colab environment recommended

2. **Required Packages**:
   - PyTorch
   - Transformers
   - datasets
   - rouge
   - lexical-diversity
   - sentencepiece
   - pandas

## Usage

1. **Initial Setup**:
   - Mount Google Drive (for saving results)
   - Verify GPU availability
   - Install required packages

2. **Data Preparation**:
   - Loads CNN/DailyMail dataset using Hugging Face's datasets library
   - Processes test split (configurable percentage)

3. **Model Loading**:
   - Loads both PEGASUS-large and PEGASUS-CNN/DailyMail models
   - Initializes corresponding tokenizers

4. **Running Experiments**:
   - Generates summaries using different decoding methods
   - Saves generated summaries to separate files
   - Computes evaluation metrics for each approach

5. **Evaluation**:
   - Calculates ROUGE scores
   - Measures lexical diversity using MTLD and HDD metrics
   - Outputs comparative results

## Configuration Parameters

- `MAX_NUM_OUTPUT_TOKENS_SMALL`: 128
- `MAX_NUM_OUTPUT_TOKENS_LARGE`: 256
- `NUM_BEAMS`: 8 (standard) and 16 (larger)
- `TOP_K`: 40 (standard) and 640 (larger)
- `TEMPERATURE`: 0.7
- `NUCLEUS_SAMPLE_VALUE`: 0.95

## Output Files

The notebook generates several output files for analysis:
- Actual reference summaries
- Generated summaries for each decoding method
- Evaluation metrics and scores

## Directory Structure

Results are saved in the following format:
```
{dataset_name}_dataset-name_{decoding_method}_{model_name}_{approach_name}_{parameters}_generated_summaries.txt
```

## Notes

- The notebook is designed to run in Google Colab for GPU access
- Progress is automatically saved to mounted Google Drive
- Evaluation metrics include both accuracy (ROUGE) and diversity measures
