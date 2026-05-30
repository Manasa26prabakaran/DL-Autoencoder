# DL- Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
<img width="767" height="240" alt="image" src="https://github.com/user-attachments/assets/e21dbbf2-bdcf-420d-a071-2576311f64fc" />


## DESIGN STEPS
### STEP 1:
Import the required libraries and load the image dataset.

### STEP 2:
Preprocess the images by normalizing and adding noise to the input images.

### STEP 3:
Build the convolutional autoencoder with encoder and decoder layers.

### STEP 4:
Compile the model using loss function and optimizer.

### STEP 5:
Train the autoencoder using noisy images as input and clean images as target.

### STEP 6:
Test the trained model and visualize the denoised output images.

## PROGRAM

### Name:P Manasa

### Register Number:212224230149

```python
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [B, 16, 14, 14]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # [B, 32, 7, 7]
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 16, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # [B, 1, 28, 28]
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Initialize model, loss function and optimizer
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    print("Name: Prathikshaa")
    print("Register Number:212224100043")

    for epoch in range(epochs):
        running_loss = 0.0

        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            # Forward pass
            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")


```

### OUTPUT

### Model Summary
<img width="667" height="421" alt="image" src="https://github.com/user-attachments/assets/d29648c6-ea1a-4b85-b629-10a23d01f1b0" />

### Training loss

## Original vs Noisy Vs Reconstructed Image
<img width="980" height="565" alt="image" src="https://github.com/user-attachments/assets/f7d9714c-fbfa-4b50-b88b-0095a8906cfc" />

## RESULT
The convolutional autoencoder model was successfully developed and trained for image denoising. The model effectively removed noise from corrupted images and reconstructed cleaner images with improved visual quality. The encoder extracted important image features, while the decoder restored the denoised images accurately. The performance of the model demonstrated that convolutional autoencoders are effective for image denoising applications.
