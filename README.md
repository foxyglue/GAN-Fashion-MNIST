# GAN-Fashion-MNIST
This project is about using GAN (Generative Adversarial Network) using fashion MNIST dataset, and then evaluate the generated image distribution with FID.

Dataset can be downloaded from:  https://github.com/zalandoresearch/fashion-mnist

On that github, donwload: 
- d
- f
- f
- f

## Process
1. Load data and scaling
    - load data and select label
    - scaling and reshape
2. Build Generator
    - **Inputs**: Takes two inputs:
      - A noise vector of shape 50 (for generating diversity).
      - A label input, which is embedded and flattened to match the noise dimensions.
      
    - **Embedding**: The label input is passed through an embedding layer to map each label to a vector space, making it easier to integrate with the noise input.
      
    - **Concatenation**: The noise and label embedding are concatenated to form a combined input for the generator.
      
    - **Dense Layers**: Several dense layers (128, 256, 512, 1024) are applied to increase the dimensionality and complexity of the generated data.
      
    - **Output Layer**: The final dense layer reshapes the data into a 28x28 image with 1 channel using a `tanh` activation to keep pixel values between -1 and 1.
    
    - **Model**: The generator is built with the concatenated input and produces a generated image as output.

3. Build Discriminator
    - **Inputs**: Takes two inputs:
      - The image input of shape 28x28x1 (grayscale image).
      - A label input, which is embedded to match the image dimensions.
    
    - **Embedding**: The label is passed through an embedding layer and reshaped to 28x28x1 so it can be combined with the image input.
    
    - **Concatenation**: The image and the reshaped label embedding are concatenated to form a single input for the discriminator.
    
    - **Flattening**: The concatenated input is flattened into a 1D vector for easier processing in fully connected layers.
    
    - **Dense Layers**: Several dense layers (512, 1024, 1024, 512) are applied to extract features from the input and perform classification.
    
    - **Output Layer**: The final dense layer has a sigmoid activation, outputting a single value between 0 and 1, which determines whether the input image is real or generated.
    
    - **Model**: The discriminator is compiled with the Adam optimizer (learning rate of 0.00005, beta_1=0.5) and binary crossentropy loss to classify real and fake images, with accuracy as the evaluation metric.

4. Combine
    - **Discriminator Frozen**: The discriminator's weights are set to non-trainable (`discriminator.trainable = False`) to ensure that only the generator is trained during the GAN training phase.
    
    - **Inputs**: 
      - The GAN takes two inputs: 
        - A noise vector (of size 50) that helps generate diverse images.
        - A label input (to condition the generation process).
    
    - **Generation**: 
      - The noise vector and label input are passed into the generator, which produces a generated image.
    
    - **Validation**: 
      - The generated image, along with the label, is passed into the discriminator. The discriminator outputs a single value (`validity`), determining if the image is real or fake.
    
    - **Model**: 
      - The GAN model combines the generator and discriminator. It takes the noise and label as input and outputs the validity (real or fake) of the generated image.
      - The GAN is compiled with the binary crossentropy loss function and uses the Adam optimizer (learning rate of 0.00005, beta_1=0.5).

5. Train
    - **Parameters**:
      - **gan_model**: The combined GAN model, used to train the generator.
      - **generator**: The generator model that creates new images.
      - **discriminator**: The discriminator model that classifies real and generated images.
      - **train_data**: The real image dataset used for training.
      - **epochs**: Number of training cycles.
      - **batch_size**: Number of samples per training batch.
      - **noise_dim**: The size of the random noise vector.
      - **total_class**: Number of classes (for label embedding).
    
    - **Helper Functions**:
      - `get_fake_sample`: Generates fake samples using random noise and labels.
      - `get_real_sample`: Fetches real images randomly from the training data.
      - `preview_generated_images`: Periodically shows a grid of generated images (every 10 epochs) to monitor progress.
    
    - **Training Loop**:
      - The training runs for the specified number of epochs.
      - In each epoch:
        - Fake samples are generated using the generator.
        - Real samples are fetched from the training data.
        - The discriminator is trained on both real and fake images, with corresponding labels. The discriminator tries to classify real images as 1 (real) and fake images as 0 (fake).
        - After training the discriminator, the generator is trained via the GAN model. The generator tries to fool the discriminator by creating images that are classified as real (label 1).
    
    - **Logging**:
      - Every 50 batches, it prints the discriminator loss and accuracy, and the generator loss for tracking progress.
    
    - **Preview**:
      - Every 10 epochs, the function shows a few generated images using `preview_generated_images`.

6. FID score
   
