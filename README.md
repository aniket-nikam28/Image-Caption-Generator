# Image-Caption-Generator
An end-to-end image captioning project using TensorFlow. The model leverages a pre-trained EfficientNet for feature extraction and a Bidirectional LSTM with Bahdanau Attention for text generation, with Beam Search for high-quality inference.

##  About The Project

This project implements a sophisticated image captioning model that bridges the gap between computer vision and natural language processing. It takes an image as input and produces a human-like, descriptive sentence that accurately describes the scene.

The entire pipeline, from data preprocessing to model training and inference, is built from scratch using TensorFlow and Keras. The model was trained on the popular **Flickr8k dataset**.

##  Key Features

-   **EfficientNetB0 Encoder**: Uses a powerful, pre-trained Convolutional Neural Network (CNN) to extract rich, high-level feature maps from images.
-   **Bidirectional LSTM Decoder**: A Recurrent Neural Network (RNN) that generates the caption word-by-word, understanding context from both past and future words.
-   **Bahdanau Attention Mechanism**: Allows the decoder to dynamically focus on the most relevant parts of the image when generating each word, leading to more accurate and context-aware captions.
-   **Beam Search Inference**: Implements a sophisticated search algorithm during prediction to generate captions that are more fluent and coherent than a simple greedy search.
-   **Interactive UI**: Includes an `ipywidgets`-based interface directly within the notebook to easily test the model with your own images.

---

##  Model Architecture

The model follows a classic encoder-decoder architecture, which is the standard for many sequence-to-sequence tasks.

1.  **Encoder**: The **EfficientNetB0** model (without its final classification layer) processes the input image and outputs a set of feature vectors. Each vector represents a different region of the image.
2.  **Attention Mechanism**: At each step of the decoding process, the attention layer calculates a set of "attention weights." These weights determine which image regions (feature vectors) are most important for predicting the next word. It then computes a context vector—a weighted average of the image features.
3.  **Decoder**: The **Bidirectional LSTM** takes the context vector from the attention layer and the previously generated word to predict the next word in the caption. This process repeats until an "end-of-sequence" token is generated.


##  Usage

After training the model (`caption_trained_model.h5`) and saving the tokenizer (`tokenizer.pkl`), you can use the provided UI in the notebook to generate captions for new images.

Alternatively, you can adapt the `predict_caption_beam` function for programmatic use:

