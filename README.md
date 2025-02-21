# NLP Project: Text Classification, Toxicity Detection & Text Generation with Streamlit

This project aims to develop a multipage Streamlit application for Natural Language Processing (NLP) tasks. The app includes three distinct pages:

1. **Polyclinic Review Classification**: A classification task for predicting the sentiment of reviews about polyclinics. The results are generated using three different models:
   - Classical ML algorithm trained on Bag of Words.
   - RNN/LSTM model with attention mechanism.
   - BERT-based model.

2. **Toxicity Detection**: A page dedicated to detecting the level of toxicity in a user-provided message using the `rubert-tiny-toxicity` model.

3. **Text Generation**: A page where users can generate text based on a given prompt using a GPT-based model. Users can control the length of the generated text, temperature, top-k, and top-p values for better customization.

## Project Structure

- **pages/**: This directory contains the Streamlit pages for different tasks.
  - `1_Polyclinic_Review_Classification.py`: Code for the classification of polyclinic reviews.
  - `2_Toxicity_Detection.py`: Toxicity detection using Rubert model.
  - `3_Text_Generation.py`: Text generation using GPT-based model.

- **models/**: Directory for storing trained models for each task (ML, RNN, BERT-based, and GPT).

- **utils/**: Contains utility functions for data processing, model training, and prediction.

## Installation

To run the project locally, follow the steps below:

1. Clone the repository:
    ```bash
    git clone https://github.com/ANDREY700/nlp_project_W10D4-5_BERT_group.git
    cd nlp_project_W10D4-5_BERT_group
    ```

2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Features

### Page 1: Polyclinic Review Classification

- **Task**: Classify reviews of polyclinics into categories such as positive, negative, or neutral.
- **Models**: 
  - A classical Machine Learning model trained using the Bag of Words technique.
  - An RNN or LSTM model with attention for sequence processing.
  - A BERT-based model for text classification.
  
### Page 2: Toxicity Detection

- **Task**: Detect the toxicity level of a given user message.
- **Model**: `rubert-tiny-toxicity`, a pre-trained Russian BERT-based model fine-tuned for toxicity classification.

### Page 3: Text Generation

- **Task**: Generate text based on user input (prompt).
- **Model**: `sberbank-ai-rugpt3-medium-based-on-gpt2`.
- **Controls**: Users can control the length of the generated text, temperature, top-k, and top-p values to affect the randomness and quality of the generated output.

## Example Usage

1. **Polyclinic Review Classification**: Users enter a review, and the app predicts whether the review is positive, negative, or neutral based on three different models.
   
2. **Toxicity Detection**: Users can enter a message, and the app detects how toxic the message is (e.g., safe, offensive).

3. **Text Generation**: Users input a prompt, and the app generates a sequence of text. You can adjust the output by changing the `max_length`, `temperature`, `top_k`, and `top_p` settings.

## Models

The models used in this project are as follows:
- **ML Model (Bag of Words)**: A traditional ML model using the Bag of Words technique.
- **RNN/LSTM with Attention**: A Recurrent Neural Network model with an attention mechanism.
- **BERT-based model**: Fine-tuned BERT model for text classification.
- **GPT Model**: Generative Pre-trained Transformer for text generation.

## Dependencies

- `streamlit`: For building the multipage app.
- `transformers`: For working with BERT-based and GPT models.
- `scikit-learn`: For the machine learning models.
- `torch`: Required for deep learning models like RNN/LSTM and BERT.
- `pandas`, `numpy`: For data manipulation.

## Contributors

- [RenaTheDv](https://github.com/RenaTheDv)
- [ANDREY700](https://github.com/ANDREY700)
- [nanzat](https://github.com/nanzat)

#### nlp_project_W10D4-5_BERT_group
nlp_project BERT_group
