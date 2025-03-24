from ..imports import *

dicti = {
  "names": [
    "Water",
    "Trees",
    "Flooded Vegetation",
    "Crops",
    "Built Area",
    "Bare Ground",
    "Snow/Ice",
    "Clouds",
    "Rangeland"
  ],
  "colors": [
    "#1A5BAB",
    "#358221",
    "#87D19E",
    "#FFDB5C",
    "#ED022A",
    "#EDE9E4",
    "#F2FAFF",
    "#C8C8C8",
    "#C6AD8D"
  ]
}

################################################################### LULC ###################################################################
def sharpen_img(img):

    # Convert the image to grayscale (optional, depending on the image and desired effect)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a sharpening kernel (adjust kernel values for different sharpening levels)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(img, -1, kernel)

    return sharpened 

TARGET_SIZE = (1024, 1024)
TARGET_SIZE = (512, 512)
TARGET_SIZE = (128, 128)
TARGET_SIZE = (256, 256)

def resize_tif_opencv(input_path, target_size=TARGET_SIZE):
    # Read the .tif file
    dataset = gdal.Open(input_path)
    bands = dataset.RasterCount  # Get number of bands
    resized_image = []

    for i in range(1, bands + 1):
        band = dataset.GetRasterBand(i).ReadAsArray()
        resized_band = cv2.resize(band, target_size, interpolation=cv2.INTER_AREA)
        resized_image.append(resized_band)

    # Combine resized bands
    resized_image = np.stack(resized_image, axis=2)
    return resized_image

    # # Write to a new .tif file
    # driver = gdal.GetDriverByName("GTiff")
    # out_ds = driver.Create(output_path, target_size[0], target_size[1], bands, gdal.GDT_Byte)
    # for i in range(1, bands + 1):
    #     out_ds.GetRasterBand(i).WriteArray(resized_image[:, :, i-1])
    # out_ds.FlushCache()
    # print(f"Resized image saved to {output_path}")


def load_image(image_path, target_size=TARGET_SIZE, normalise=True, preprocess=True):
    img = resize_tif_opencv(image_path, target_size)
    if preprocess:
      # img = esri_img(img)
      # img = brovey_img(img)
      img = sharpen_img(img)
      # pass
    # img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)  # numpy array
    if normalise:
      img = img / 255.0  # Normalize to [0, 1]
    return img

def load_mask(image_path, target_size=TARGET_SIZE):
    # Resize
    img = resize_tif_opencv(image_path, target_size)
    # img = load_img(image_path, target_size=target_size, color_mode='grayscale')
    img = img_to_array(img)  # numpy array
    img = img.astype(np.uint8)  # Ensure the mask is integer type

    # One-hot encode the mask (1 to 9 -> 9 classes)
    mask = np.zeros((img.shape[0], img.shape[1], 9), dtype=np.uint8)
    for i in range(1, 10):  # Classes are from 1 to 9
        # mask[..., i-1] = (img == i).astype(np.uint8)
        mask[:, :, i-1] = (img[:, :, 0] == i).astype(np.uint8)  # Select the correct channel
    return mask


def fetch_sat_mask_dir(place_name, sat_dir):
    sat_image_path = os.path.join(sat_dir, f"Satellite_{place_name}.tif")

    if os.path.exists(sat_image_path):
        satellite_image = load_image(sat_image_path, target_size=TARGET_SIZE, normalise=True, preprocess=True)

    return satellite_image


def extract_lulc_features(place_name, sat_dir, model_file, output_mask_dir):
  # try:
    # Assuming X_test and y_test are preprocessed and correctly shaped for your models.
    X_data = fetch_sat_mask_dir(place_name, sat_dir)
    if X_data is None:
      return None

    with open(model_file, 'rb') as f:
      model = pickle.load(f)

    pred_mask = model.predict(np.expand_dims(X_data, axis=0),  verbose=0)[0]
    print(pred_mask.shape)
    pred_mask = np.argmax(pred_mask, axis=-1) + 1  # Convert one-hot to class predictions

    # # Plot predicted mask
    # cmap = plt.cm.colors.ListedColormap(dicti["colors"])
    # bounds = np.arange(1,11) # Define the bounds for the colors (1 to 9 plus one extra for the colorbar)
    # norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
    # plt.imshow(pred_mask, cmap=cmap, norm=norm)



    # Create a colored image
    colored_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for class_value, color_hex in enumerate(dicti["colors"], start=1):
        # Convert hex color to RGB
        color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
        # Apply color to corresponding pixels
        colored_mask[pred_mask == class_value] = color_rgb

    pred_mask_image = Image.fromarray(colored_mask)
    output_path = f"{output_mask_dir}/lulc_mask_{place_name}.png"
    pred_mask_image.save(output_path)

    total_pixels = pred_mask.size
    ratios = {}

    # Calculate the proportion for each unique value
    unique_values, counts = np.unique(pred_mask, return_counts=True)
    for num in range(1, 10):
      if num not in unique_values:
        unique_values = np.append(unique_values, num)
        counts = np.append(counts, 0)

    # print(unique_values)
    # print(counts)
    for value, count in zip(unique_values, counts):
        ratios[f"{value}_ratio"] = count / total_pixels
    print(ratios)
    return ratios

  # except Exception as e:
  #   print(f"Error loading or predicting with model {model_file}: {e}")
  #   # Handle the error as needed (e.g., skip the model)