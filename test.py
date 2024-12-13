import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms


def apply_segmentation_model(image_path, model, transform, device):
    """
    Apply a segmentation model to an input image and return the resulting segmentation mask.

    Parameters:
    - image_path (str): Path to the input image.
    - model: Pretrained segmentation model.
    - transform: Preprocessing transformation pipeline for the image.
    - device: The device to perform computations on (CPU or GPU).

    Returns:
    - np.ndarray: Segmentation mask as a NumPy array.
    - PIL.Image: Original input image for visualization.
    """
    # Load and preprocess the image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {e}")
    
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(input_tensor).logits
        predictions = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()

    return predictions, image


# Example Usage
if __name__ == "__main__":
    # Model and preprocessing setup
    from transformers import SegformerForSemanticSegmentation

    # Load the model
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

    # Customize the model for 3 classes
    num_classes = 3
    model.decode_head.classifier = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))
    model.config.num_labels = num_classes

    # Load custom checkpoint
    checkpoint = torch.load("final_model.pth", map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Define device and preprocessing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply the function to a test image
    test_image_path = "./static/uploads/doctor.png"
    mask, original_image = apply_segmentation_model(test_image_path, model, transform, device)

    # Visualize the results
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(mask, cmap="jet")
    plt.axis("off")

    plt.show()
