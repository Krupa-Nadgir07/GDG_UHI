from ..imports import *

def get_latent_representation(image_path, model_dir_prithivi, image_size=(256, 256)):
    """
    Takes an image path as input, processes the image, and outputs the latent representation.

    Parameters:
        image_path (str): Path to the image file.
        image_size (tuple): The target size for resizing the image.

    Returns:
        np.ndarray: Latent representation of the input image.
    """
    try:
        # Load the pre-trained autoencoder model and encoder model
        # Ensure these are saved as 'autoencoder.h5' and 'encoder.h5' respectively
        autoencoder = load_model(f'{model_dir_prithivi}/final_model/autoencoder.h5')
        encoder = load_model(f'{model_dir_prithivi}/final_model/encoder.h5')
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image at path {image_path} could not be loaded.")

        # Preprocess the image
        img = cv2.resize(img, image_size)  # Resize image
        img = img.astype('float32') / 255.0  # Normalize pixel values to [0, 1]
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Get the latent representation
        latent_vector = encoder.predict(img)
        latent_vector = latent_vector.flatten()  # Flatten the latent vector

        return latent_vector
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def fetch_uhi(image_path, model_dir_prithivi):
  # image_path = '/Users/prithivi/Desktop/EL/first model/indian_images/2020_04_5.tif'  # Change to your image path
  latent_representation = get_latent_representation(image_path, model_dir_prithivi)

  if latent_representation is not None:
      print("Latent Representation:")
      print(latent_representation)
  else:
      print("Failed to generate latent representation.")
  uhi=load_model(f"{model_dir_prithivi}/final_model/model.h5")

  # Expand dimensions to add the batch size (e.g., from (32,) to (1, 32))
  latent_representation = np.expand_dims(latent_representation, axis=0)

  # Perform prediction
  uhi_index_prediction = uhi.predict(latent_representation)

  # Since the output will still have a batch dimension, squeeze it if you want a scalar value
  uhi_index_prediction = np.squeeze(uhi_index_prediction)

  print(f"Predicted UHI index: {uhi_index_prediction}")
  return uhi_index_prediction