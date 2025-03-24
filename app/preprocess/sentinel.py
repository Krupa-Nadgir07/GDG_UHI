from ..imports import *

################################################################### Earth Engine ###################################################################
# Function to create a bounding box ROI
def createROI(lat, lon, boxSize):
    halfSize = ee.Number(boxSize).divide(2)
    minlat = ee.Number(lat).subtract(halfSize)
    maxlat = ee.Number(lat).add(halfSize)
    minlon = ee.Number(lon).subtract(halfSize)
    maxlon = ee.Number(lon).add(halfSize)
    return ee.Geometry.Rectangle([minlon, minlat, maxlon, maxlat])


################################################################### Sentinel Features ###################################################################

def extract_sentinel_features(lat, lon, year, place_name):
  try:
    retrieved_img = True
    ##################################################################
    # Create ROI
    boxSize = 0.1
    roi = createROI(lat, lon, boxSize)

    start_date = f'{year}-02-01'
    end_date = f'{year}-10-30'  # Fixed date range for simplicity

    ##################################################################
    # Define cloud cover thresholds
    cloud_thresholds = [3, 10, 20]  # From strict to lenient
    cd_thresh = 3
    selected_image = None

    for threshold in cloud_thresholds:
        # Load Sentinel-2 image collection with the current threshold
        image_collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', threshold))
            .filterBounds(roi)
        )

        if image_collection.size().getInfo() > 0:
            selected_image = image_collection.median()  # Composite of images
            # selected_image = image_collection.first()
            # print(f"Image selected for {city_name} ({urban_id}) with cloud threshold {threshold}%.")
            cd_thresh = threshold
            break

    if not selected_image:
        retrieved_img = False
        return None

    ##################################################################
    # Check available bands
    available_bands = selected_image.bandNames().getInfo()
    # print(f"Available bands for {city_name} ({urban_id}): {available_bands}")

    # Select visualization bands for RGB (exclude QA60)
    vis_bands = ['B4', 'B3', 'B2'] if all(b in available_bands for b in ['B4', 'B3', 'B2']) else available_bands[:3]

    # Ensure 'QA60' exists for cloud masking
    if 'QA60' not in available_bands or len(vis_bands) < 3:
        # print(f"QA60 band is not available for {city_name} ({urban_id}). Skipping...")
        retrieved_img = False
        return None

    ##################################################################
    # Function to mask clouds using QA60
    def mask_s2_clouds(image):
        # cloud_mask = image.select('QA60').bitwiseAnd(1 << 10).eq(0)  # Cloud-free pixels
        # return image.updateMask(cloud_mask)
        qa60 = image.select('QA60').int()
        cloud_mask = qa60.bitwiseAnd(1 << 10).eq(0)  # Cloud-free pixels
        return image.updateMask(cloud_mask)

    # Apply cloud masking to the selected image
    cloud_masked_image = mask_s2_clouds(selected_image)

    ##################################################################
    # Check the number of valid pixels after cloud masking
    valid_pixel_count = cloud_masked_image.reduceRegion(
        reducer=ee.Reducer.count(),
        geometry=roi,
        scale=10,
        maxPixels=1e13
    ).get('B4')  # Use any band available, e.g., 'B4'

    # Fetch the count of valid pixels as an integer
    valid_pixel_count = valid_pixel_count.getInfo() if valid_pixel_count else 0
    # print("Number of valid_pixel_count:", valid_pixel_count)

    # Skip export if the valid pixel count is below a threshold
    if valid_pixel_count < 1200000:  # Threshold: 1240996 --- 1239882
      retrieved_img = False
      return None

    ##################################################################
    # Visualize the cloud-masked image (RGB only)
    image2 = cloud_masked_image.visualize(bands=vis_bands, min=0, max=2500, gamma=1.1)

    deploymnet_dir = "G:/My Drive/Deployment_EarthEngineExports"
    sat_file = f"Satellite_{place_name}.tif"
    sat_path = os.path.join(deploymnet_dir,sat_file)

     # Caching check for satellite image
    if not os.path.exists(sat_path):
        task2 = ee.batch.Export.image.toDrive(
            image=image2,
            description=f'Satellite_{place_name}',
            fileNamePrefix=f'Satellite_{place_name}',
            scale=10,
            region=roi,  # Pass the GeoJSON directly
            fileFormat='GeoTIFF',
            folder=f'Deployment_EarthEngineExports',
            maxPixels=1e13
        )
        task2.start()
        while task2.active():
            print(f"Task for {place_name} is running...")
            time.sleep(10)

        print(f"Export completed for {place_name} ")

    ##################################################################
    # Extract numerical satellite features
    image = selected_image
    available_bands = image.bandNames().getInfo()
    # print(available_bands)

    if "B8" in available_bands and "B4" in available_bands:
        ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI").toFloat()
    else:
        ndvi = None

    if "B8" in available_bands and "B12" in available_bands:
        nbr = image.normalizedDifference(["B8", "B12"]).rename("NBR").toFloat()
    else:
        nbr = None

    if "B11" in available_bands and "B8" in available_bands:
        ndui = image.normalizedDifference(["B11", "B8"]).rename("NDUI").toFloat()
        ndbi = image.normalizedDifference(["B11", "B8"]).rename("NDBI").toFloat()
    else:
        ndui = None
        ndbi = None

    if "B11" in available_bands and "B12" in available_bands and "B8" in available_bands:
        nmdi = image.expression(
            "((SWIR1 - (NIR + SWIR2)) / (SWIR1 + (NIR + SWIR2)))",
            {
                "SWIR1": image.select("B11"),
                "NIR": image.select("B8"),
                "SWIR2": image.select("B12")
            },
        ).rename("NMDI").toFloat()

        # Define bands for calculations
        NIR = image.select('B8')   # Near-Infrared
        RED = image.select('B4')   # Red
        BLUE = image.select('B2')  # Blue
        SWIR1 = image.select('B11')  # Shortwave Infrared 1
        SWIR2 = image.select('B12')  # Shortwave Infrared 2

        # Compute Soil-Adjusted Vegetation Index (SAVI)
        # Formula: SAVI = ((NIR - RED) * (1 + L)) / (NIR + RED + L), where L = 0.5
        L = 0.5
        savi = image.expression(
            '((NIR - RED) * (1 + L)) / (NIR + RED + L)',
            {'NIR': NIR, 'RED': RED, 'L': L}
        ).rename('SAVI').toFloat()

        # Compute Urban Heat Surface Index (UHSI)
        # Proxy using SWIR bands
        uhsi = image.expression(
            'SWIR1 / SWIR2',
            {'SWIR1': SWIR1, 'SWIR2': SWIR2}
        ).rename('UHSI').toFloat()

    else:
        nmdi = None
        savi = None
        uhsi = None

    if ndbi and ndvi:  # Ensure both indices exist before combining
        shade = ndvi.multiply(-1).add(1).rename("Shade").toFloat()
        ndbsi = ndbi.add(shade).divide(2).rename("NDBSI").toFloat()
    else:
        shade = None
        ndbsi = None

    if "B12" in available_bands and "B3" in available_bands:
        nbadi = image.normalizedDifference(["B12", "B3"]).rename("NBaDI").toFloat()
    else:
        nbadi = None

    if "WVP" in available_bands:
        wvp = image.select("WVP").toFloat()
    else:
        wvp = None

    if "AOT" in available_bands:
        aot = image.select("AOT").toFloat()
    else:
        aot = None

    if "B11" in available_bands:
        lst = image.select("B11").rename("LST").toFloat()
    else:
        lst = None

    if ("B3" in available_bands):
        gndvi = image.normalizedDifference(["B8", "B3"]).rename("GNDVI").toFloat()
        ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI").toFloat()
    else:
        gndvi = None
        ndwi = None



    # indices = ndvi.addBands([nbr, ndui, gndvi, ndwi, shade, ndbi, nmdi, ndbsi, nbadi, wvp, aot, lst, savi, uhsi])

    # Calculate statistics for each band
    stat_names = ["min", "max", "mean", "stdDev"]
    stats_dict = {}

    ##################################################################
    # Combine indices into a single image only if they are not None
    indices = ee.Image()
    for category in [ndvi, nbr, ndui, gndvi, ndwi, shade, ndbi, nmdi, ndbsi, nbadi, wvp, aot, lst, savi, uhsi]:
      if category is None:
        stats_dict[f"{category}_mean"] = None
        stats_dict[f"{category}_stdDev"] = None
        stats_dict[f"{category}_min"] = None
        stats_dict[f"{category}_max"] = None
      else:
        print(type(category))
        indices = indices.addBands([category.toFloat()])
    # print(indices)

    # Caching check for features image
    deploymnet_dir = "G:/My Drive/Deployment_EarthEngineExports"
    feat_file = f"Features_{place_name}.tif"
    feat_path = os.path.join(deploymnet_dir,feat_file)

    if not os.path.exists(feat_path):
        task1 = ee.batch.Export.image.toDrive(
            image=indices.toFloat(),
            description=f'Features_{place_name}',
            fileNamePrefix=f'Features_{place_name}',
            scale=10,
            region=roi,
            fileFormat='GeoTIFF',
            folder=f'Deployment_EarthEngineExports',
            maxPixels=1e13
        )
        task1.start()

    ##################################################################
    for band in indices.bandNames().getInfo():
        stats = indices.select(band).reduceRegion(
            reducer=ee.Reducer.minMax()
            .combine(ee.Reducer.mean(), sharedInputs=True)
            .combine(ee.Reducer.stdDev(), sharedInputs=True),
            geometry=roi,
            scale=10,
            maxPixels=1e13
        )
        time.sleep(0.3)
        stats_dict.update(stats.getInfo())


    # Wait for task to complete
    # while task2.active() or task1.active():
    

    return stats_dict

  except Exception as e:
    print(f"Error processing {place_name} : {e}")
