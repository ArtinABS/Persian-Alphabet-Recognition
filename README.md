<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
</head>
<body>

<h1>Persian Alphabet Recognition</h1>
<p><em>Efficient recognition of Persian handwritten alphabet letters using a variety of optimized machine learning models.</em></p>

<h2>Project Overview</h2>
<p>This repository provides a comprehensive system for recognizing Persian handwritten alphabet letters. Leveraging multiple optimized models, it aims to achieve high accuracy in character recognition. The project includes datasets, model implementations, and testing tools, organized for easy navigation and integration.</p>

<h2>Directory Structure</h2>
<ul>
  <li><strong>Datasets/</strong> - Contains labeled images of Persian characters, used for training and evaluation.</li>
  <li><strong>Models/</strong> - Includes pre-trained model files and architectures optimized for character recognition tasks.</li>
</ul>

<h2>Models Preview</h2>
<p>This project implements a range of models and ensemble techniques to maximize accuracy in Persian handwritten character recognition:</p>

<ul>
  <li><strong>Artificial Neural Network (ANN)</strong> - A fully connected neural network for baseline character recognition.</li>
  <li><strong>Convolutional Neural Network (CNN)</strong> - Specialized in handling image data, achieving high accuracy with character spatial hierarchies.</li>
  <li><strong>K-Nearest Neighbors (KNN)</strong> - Classifies characters based on their feature similarity with neighbors.</li>
  <li><strong>Support Vector Machine (SVM)</strong> - Utilizes optimal hyperplane boundaries to classify character data.</li>
  <li><strong>Decision Tree</strong> - A tree-based model that splits data based on feature values for classification.</li>
  <li><strong>Logistic Regression</strong> - A linear model for binary classification, used as a simpler baseline.</li>
  <li><strong>Ensemble Learning</strong>:
    <ul>
      <li><strong>Bagging</strong> - Combines multiple models to reduce variance and improve accuracy.</li>
      <li><strong>Boosting</strong> - Sequentially trains models to minimize errors from previous models.</li>
      <li><strong>Stacking</strong> - Stacks combinations of models (e.g., CNN, SVM, Decision Trees) to boost performance by leveraging diverse predictions.</li>
    </ul>
  </li>
</ul>

<h2>Requirements</h2>
<p>To replicate this project, you will need the following:</p>
<ul>
  <li>Python 3.x</li>
  <li>Jupyter Notebook</li>
  <li>Machine learning libraries: TensorFlow, scikit-learn, NumPy, etc.</li>
</ul>

<h2>Getting Started</h2>
<p>Clone the repository and navigate to the project directory:</p>
<pre><code>git clone https://github.com/ArtinABS/Persian-Alphabet-Recognition.git
cd Persian-Alphabet-Recognition
</code></pre>

<h2>Usage</h2>
<p>Open Models folder to view the training process and evaluate model performance. You can use the datasets provided in <code>Datasets/</code> to test additional models or fine-tune existing ones.</p>

<h2>License</h2>
<p>This project is licensed under the MIT License. See the <a href="LICENSE">LICENSE</a> file for more information.</p>


<h2>Contributors</h2>
<p>Special thanks to everyone who contributed to this project.</p>

</body>
</html>
