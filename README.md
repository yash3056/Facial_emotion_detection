# Facial Emotion Recognition

Facial Emotion Recognition is a deep learning project that aims to detect and classify human emotions from facial expressions. This project leverages multiple models, including Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs), to achieve high accuracy in recognizing different emotions.

## Why Facial Emotion Detection?

Facial emotion detection is a critical technology in various applications such as human-computer interaction, mental health monitoring, marketing analysis, and security systems. By accurately interpreting human emotions, machines can provide more personalized and effective responses, enhance user experience, and improve the efficiency of numerous automated systems.

## Dataset

This project uses the FERPlus dataset, which consists of images representing 8 different emotions:

- Angry
- Contempt
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

The dataset is divided into three parts:

- **Train**: 66,379 samples
- **Test**: 3,579 samples
- **Validation**: 8,341 samples

To address class imbalance, various data augmentation techniques were applied to ensure a balanced dataset. You can find the dataset [here](https://www.kaggle.com/datasets/arnabkumarroy02/ferplus).

## Models

This project explores four different models for facial emotion recognition:

1. **CNN with 4 Conv2D Layers**: This model uses a simple architecture with four convolutional layers to extract features from facial images and classify emotions. It is suitable for quick experimentation and baseline comparisons.  
   [Notebook link](https://github.com/yash3056/Facial_emotion_detection/blob/main/Emotion%20Analysis%20using%204%20Conv2d%20layers.ipynb)

2. **ResNet50**: A deeper convolutional neural network, ResNet50, leverages residual learning to train effectively on large datasets. It provides improved accuracy over simpler CNN models by using skip connections to mitigate the vanishing gradient problem.  
   [Notebook link](https://github.com/yash3056/Facial_emotion_detection/blob/main/Emotion%20Analysis%20Using%20ResNet50.ipynb)

3. **Vision Transformer (ViT) - Version 1**: This model uses the Vision Transformer architecture with the pre-trained model 'google/vit-base-patch16-224-in21k'. It applies transformer techniques, originally designed for NLP, to image classification tasks, offering excellent performance on various benchmarks.  
   [Notebook link](https://github.com/yash3056/Facial_emotion_detection/blob/main/ferplus-vit_V_1.ipynb)

4. **Vision Transformer (ViT) - Version 2**: Another Vision Transformer model, but with the 'microsoft/beit-base-patch16-224-pt22k-ft22k' pre-trained model. This version explores different transformer architectures and training strategies to achieve high accuracy in emotion detection.  
   [Notebook link](https://github.com/yash3056/Facial_emotion_detection/blob/main/ferplus-vit_V_2.ipynb)

## Installation

To run this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/yash3056/Facial_emotion_detection.git
cd Facial_emotion_detection
pip install -r requirements.txt
```

Ensure that you have the necessary libraries installed, such as TensorFlow or PyTorch, depending on the model you're using.

## Usage

After installing the dependencies, you can run any of the provided Jupyter notebooks to train and evaluate the models. Simply open the notebook in Jupyter and run the cells sequentially.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- [Yash Prakash Narayan](https://github.com/yash3056)

## Acknowledgments

- The FERPlus dataset from [Kaggle](https://www.kaggle.com/datasets/arnabkumarroy02/ferplus)
- These Models where trained on Kaggle and colab
- Pre-trained models from the Hugging Face library

## Limitation

Due to hardware limitation better models cannot be used with fusion of deep learning algorithm
