# ASL Recognition Model

This project provides an open-source implementation of an American Sign Language (ASL) Recognition Model. The model leverages machine learning and computer vision techniques to recognize ASL hand signs from images.

## Features

- **Hand Landmark Detection**: Utilizes MediaPipe to accurately detect 21 hand landmarks in images.
- **Feature Extraction**: Calculates angles between all pairs of landmarks to form a 420-dimensional feature vector.
  - **Vector Calculation**: Computes vectors between each pair of landmarks.
  - **Angle Computation**: Uses the arccosine of normalized vector components to derive angles.
- **Model Input**: The extracted angles serve as input features for the Random Forest model, which classifies the ASL sign.

## Technical Stack

- **Python**: Core programming language.
- **OpenCV**: For image processing and manipulation.
- **MediaPipe**: For detecting hand landmarks.
- **Scikit-learn**: Provides the Random Forest model for classification.
- **Streamlit**: Facilitates an interactive user interface for real-time recognition.

## Supported Alphabets

The model currently works for the following ASL alphabets:
- A, B, C, E, F, G, H, I, J, K, L, O, Q, R, S, W, Y

The model does not support or may not work correctly for:
- D, M, N, P, T, U, V, X, Z

## Usage

1. Upload an image of an ASL sign through the Streamlit interface.
2. The model processes the image and provides the top 5 predictions along with visualizations of detected hand landmarks.

## Contribution

We welcome contributions to improve the model's accuracy and expand its alphabet coverage. Feel free to fork the repository, submit issues, or create pull requests.

## License

This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgments

Thanks to the contributors of MediaPipe and Scikit-learn for their powerful libraries that made this project possible.