from ..imports import *
################################################################### Socio - Urban Features ###################################################################

start_date = '2024-01-01'
end_date = '2024-08-01'
pop_path = "G:/My Drive/5th_Sem_EL/LULC_Datasets/Load_Dataset/GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0.tif"
settlement_path = 'G:/My Drive/5th_Sem_EL/LULC_Datasets/Load_Dataset/GHS_BUILT_S_E2020_GLOBE_R2023A_4326_3ss_V1_0.tif'
built_height_path = 'G:/My Drive/5th_Sem_EL/LULC_Datasets/Load_Dataset/GHS_BUILT_H_AGBH_E2018_GLOBE_R2023A_4326_3ss_V1_0.tif'
morph_file = 'G:/My Drive/5th_Sem_EL/LULC_Datasets/Load_Dataset/GHS_BUILT_C_MSZ_E2018_GLOBE_R2023A_54009_10_V1_0.tif'
gisd30 = ee.Image("projects/sat-io/open-datasets/GISD30_1985_2020")
ghap = ee.ImageCollection("projects/sat-io/open-datasets/GHAP/GHAP_M1K_PM25")
no2_dataset = ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2")

#Degree of urbanisation

def get_degree_of_urbanization(lat, lon):
    """Extracts Degree of Urbanization index from JRC GHS SMOD dataset."""
    try:
        point = ee.Geometry.Point(lon, lat)
        image = ee.Image("JRC/GHSL/P2023A/GHS_SMOD_V2-0/2030")
        smod = image.select('smod_code')
        result = smod.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=30,  # Scale in meters
            maxPixels=1e8
        ).getInfo()
        return result.get('smod_code', None)
    except Exception as e:
        print(f"Error fetching Degree of Urbanization for {lat}, {lon}: {e}")
        return None

#Night Light Intensity
def get_night_light_intensity(lat, lon):
    """Extracts night light intensity from VIIRS dataset."""
    try:
        point = ee.Geometry.Point(lon, lat)
        dataset = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG") \
            .filterDate(start_date, end_date).mean()
        result = dataset.reduceRegion(ee.Reducer.mean(), point, 1000).getInfo()
        return result.get('avg_rad', None)
    except Exception as e:
        print(f"Error fetching night light intensity for {lat}, {lon}: {e}")
        return None
    
# Population Density

# Function to calculate grid cell area at a given latitude
def calculate_grid_cell_area(lat):
    earth_radius = 6371.0  # Earth's radius in km
    angular_resolution = 3 / 3600.0  # Resolution in degrees (3 arc-seconds)
    lat_rad = math.radians(lat)
    cell_width = (angular_resolution / 360) * (2 * math.pi * earth_radius * math.cos(lat_rad))
    cell_height = (angular_resolution / 360) * (2 * math.pi * earth_radius)
    return cell_width * cell_height  # Area in km²

# Function to calculate the mean population density in a given radius
def get_density_in_radius(lat, lon, radius_km=15):
    try:
        with rasterio.open(pop_path) as raster:
            step = 0.005  # Approx. 0.55 km in degrees
            densities = []

            for lat_offset in np.arange(-radius_km / 111, radius_km / 111, step):
                for lon_offset in np.arange(-radius_km / 111, radius_km / 111, step):
                    test_lat = lat + lat_offset
                    test_lon = lon + lon_offset
                    for val in raster.sample([(test_lon, test_lat)]):
                        if val[0] is not None and val[0] > 0:  # Only valid positive values
                            grid_cell_area = calculate_grid_cell_area(test_lat)
                            densities.append(val[0] / grid_cell_area)
            return np.mean(densities) if densities else None
    except Exception as e:
        print(f"Error in radius-based density calculation: {e}")
        return None

# Function to extract population density from the raster file or use fallbacks
def get_population_density(lat, lon):
    try:
        # Attempt direct sampling from the raster file
        with rasterio.open(pop_path) as raster:
            coord = (lon, lat)  # (longitude, latitude)
            for val in raster.sample([coord]):
                if val[0] is not None and val[0] > 0:  # Valid positive population value
                    grid_cell_area = calculate_grid_cell_area(lat)
                    return val[0] / grid_cell_area

        # Adaptive Radius Fallback: Start at 15 km, expand to 40 km
        for radius_km in range(15, 41, 5):  # Increase by 5 km increments
            density = get_density_in_radius(lat, lon, radius_km=radius_km)
            if density is not None:
                return density

        # Final Fallback: Calculate global mean density within 75 km radius
        global_density = get_density_in_radius(lat, lon, radius_km=75)
        if global_density is not None:
            return global_density

    except Exception as e:
        # Log the error (optional)
        print(f"Error calculating population density: {e}")

# Built Settlement Area

# Function to calculate grid cell area at a given latitude
def calculate_grid_cell_area(lat):
    base_area = 0.0081  # Base area (km²) for 3 arcsecond resolution at the equator
    return base_area * math.cos(math.radians(lat))  # Adjust for latitude

# Function to calculate the mean built-up fraction in a given radius
def get_mean_built_up_in_radius(lat, lon, radius_km=15):
    try:
        step_km = 0.5  # Desired spatial resolution in km
        lat_step = step_km / 111  # Convert step to degrees for latitude
        lon_step = step_km / (111 * math.cos(math.radians(lat)))  # Convert step to degrees for longitude

        bufracs = []
        with rasterio.open(settlement_path) as raster:
            for lat_offset in np.arange(-radius_km / 111, radius_km / 111, lat_step):
                for lon_offset in np.arange(-radius_km / (111 * math.cos(math.radians(lat))),
                                             radius_km / (111 * math.cos(math.radians(lat))),
                                             lon_step):
                    test_lat = lat + lat_offset
                    test_lon = lon + lon_offset
                    for val in raster.sample([(test_lon, test_lat)]):
                        if val[0] is not None and val[0] > 0:  # Only consider valid, positive values
                            bufracs.append(val[0])
        return np.mean(bufracs) if bufracs else None  # Return None if no valid values are found
    except Exception as e:
        print(f"Error processing radius for ({lat}, {lon}): {e}")
        return None

# Function to extract built-up area from the raster file and handle nulls
def get_built_area_km2(lat, lon):
    try:
        # Open the raster file
        with rasterio.open(settlement_path) as raster:
            # Convert latitude and longitude to raster coordinates
            coord = (lon, lat)  # rasterio uses (longitude, latitude)

            # Sample the raster at the specified coordinate
            for val in raster.sample([coord]):
                if val[0] is not None:  # Check for valid BUFRAC value
                    # Calculate the area of the grid cell at the given latitude
                    grid_cell_area = calculate_grid_cell_area(lat)
                    # Calculate the built-up area (BUFRAC * cell area)
                    built_up_area = val[0] * grid_cell_area
                    if built_up_area > 0:
                        return built_up_area  # Return direct built-up area if non-zero

        # Adaptive Radius Fallback: Start at 15 km, expand to 40 km
        for radius_km in range(15, 41, 5):  # Increase by 5 km increments
            mean_built_up_fraction = get_mean_built_up_in_radius(lat, lon, radius_km)
            if mean_built_up_fraction is not None:
                # Convert fraction to area by multiplying by a grid cell area
                grid_cell_area = calculate_grid_cell_area(lat)
                return mean_built_up_fraction * grid_cell_area

        # Final Fallback: Calculate global mean fraction within 75 km radius
        global_mean_fraction = get_mean_built_up_in_radius(lat, lon, radius_km=75)
        if global_mean_fraction is not None:
            grid_cell_area = calculate_grid_cell_area(lat)
            return global_mean_fraction * grid_cell_area

    except Exception as e:
        print(f"Error processing coordinate ({lat}, {lon}): {e}")

#Built Settlement Height


# Function to calculate the mean height in a given radius
def get_mean_height_in_radius(lat, lon, radius_km):
    try:
        with rasterio.open(built_height_path) as raster:
            transform = raster.transform  # Raster transform
            radius_deg = radius_km / 111  # Approximate conversion from km to degrees
            window_size = int(radius_deg * 3600 / 3)  # Convert radius to pixels at 3 arcsecond resolution

            # Convert lat/lon to raster indices
            lat_idx, lon_idx = raster.index(lon, lat)

            # Extract a window around the location
            window = raster.read(1, window=((max(lat_idx - window_size, 0), min(lat_idx + window_size, raster.height)),
                                             (max(lon_idx - window_size, 0), min(lon_idx + window_size, raster.width))))
            # Filter out nodata values
            valid_values = window[window > 0]
            return np.mean(valid_values) if valid_values.size > 0 else None
    except Exception as e:
        print(f"Error calculating mean height for ({lat}, {lon}): {e}")
        return None

# Function to extract built height (AGBH) from the raster file
def get_height(lat, lon):
    try:
        with rasterio.open(built_height_path) as raster:
            coord = (lon, lat)  # rasterio uses (longitude, latitude)
            for val in raster.sample([coord]):
                if val[0] is not None and val[0] > 0:  # Valid height value in meters
                    return val[0]
    except Exception as e:
        print(f"Error processing coordinate ({lat}, {lon}): {e}")
    return None

# Function to compute built height with adaptive fallback handling
def get_built_height(lat, lon):
    try:
        # Step 1: Try direct extraction
        height = get_height(lat, lon)
        if height is not None and height > 0:
            return height

        # Step 2: Adaptive radius fallback (15 to 40 km)
        for radius_km in range(15, 41, 5):
            mean_height = get_mean_height_in_radius(lat, lon, radius_km)
            if mean_height is not None:
                return mean_height

        # Step 3: Global mean fallback within a 75 km radius
        global_mean_height = get_mean_height_in_radius(lat, lon, radius_km=75)
        if global_mean_height is not None:
            return global_mean_height

    except Exception as e:
        print(f"Error in built height computation for ({lat}, {lon}): {e}")



# Morphological Settlement Zone Classification
# Function to calculate distance between two points (in km)
def calculate_distance(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

# Function to get the mode of a list
def calculate_mode(values):
    try:
        return stats.mode(values)[0][0]
    except IndexError:
        return None

# Function to extract MSZ value based on lat, lon only
def extract_msz_value(lat, lon):
    radius_km_start = 0
    radius_km_step = 15
    max_radius_km = 20

    # Open the raster file
    with rasterio.open(morph_file) as src:
        raster_crs = src.crs  # Get raster CRS
        nodata_value = src.nodata  # Get NoData value from the raster
        transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)

        # Get raster bounds
        left, bottom, right, top = src.bounds

        # Transform WGS84 coordinates to raster CRS
        x, y = transformer.transform(lon, lat)

        # Ensure transformed coordinates are within raster bounds
        if not (left <= x <= right and bottom <= y <= top):
            return None  # Skip out-of-bounds points

        # Get row and column indices for the raster
        try:
            row_idx, col_idx = src.index(x, y)
        except (IndexError, ValueError):
            return None  # If coordinates are invalid, return None

        # Extract the value from the raster at the given indices
        window = rasterio.windows.Window(col_idx, row_idx, 1, 1)  # Define a single-pixel window
        value = src.read(1, window=window)[0, 0]  # Read the single-pixel value

        # If the value is 0, 255, or NoData, calculate the mode of nearby valid values
        if value == 0 or value == 255 or value == nodata_value:
            found_valid_value = False
            for radius_km in range(radius_km_start, max_radius_km + 1, radius_km):
                nearby_values = []
                # Check nearby points in the radius
                for other_lat in range(int(lat-radius_km), int(lat+radius_km)):
                    for other_lon in range(int(lon-radius_km), int(lon+radius_km)):
                        distance = calculate_distance(lat, lon, other_lat, other_lon)
                        if distance <= radius_km:
                            other_x, other_y = transformer.transform(other_lon, other_lat)
                            try:
                                other_row_idx, other_col_idx = src.index(other_x, other_y)
                                other_window = rasterio.windows.Window(other_col_idx, other_row_idx, 1, 1)
                                other_value = src.read(1, window=other_window)[0, 0]
                                if other_value not in [0, 255, nodata_value]:
                                    nearby_values.append(other_value)
                            except (IndexError, ValueError):
                                continue

                if nearby_values:
                    value = calculate_mode(nearby_values)
                    found_valid_value = True
                    break

            if not found_valid_value:
                value = None

        return value

#Global Impervious Surface Area
def get_gisd30_value(lat, lon):
    """Extracts GISD30 value at a specific location, calculates the mode if the value is 0."""
    try:
        point = ee.Geometry.Point(lon, lat)
        result = gisd30.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=point,
            scale=30
        ).getInfo()

        value = result.get('b1') if result else 0

        if value == 0:
            # If value is 0, calculate the mode within a 15 km radius
            buffer = point.buffer(15 * 1000)  # 15 km buffer in meters
            values = [feature.get('b1') for feature in gisd30.sampleRegions(
                collection=ee.FeatureCollection([ee.Feature(buffer)]),
                scale=30, geometries=True
            ).getInfo()['features']]

            values = [v for v in values if v is not None]
            value = Counter(values).most_common(1)[0][0] if values else 0

        return value
    except:
        return 0

#PM2.5 concentration
# Define the scaling function
def scale(image):
    return image.multiply(0.1).copyProperties(image, ['system:time_start', 'system:time_end'])

# Apply scaling globally
scaled_dataset = ghap.map(scale)

# Define a function to get PM2.5 value for a given latitude and longitude
def get_pm25_value(lat, lon):
    point = ee.Geometry.Point(lon, lat)
    pm25_image = scaled_dataset.filterDate('2020-01-01', '2020-12-31').mean()

    try:
        # Get the PM2.5 value at the specified point
        value = pm25_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=1000
        ).get('b1')  # Use the correct band name 'b1'

        # If the value is None, check within a 15 km radius
        if value is None:
            # Create a buffer of 15 km around the point
            buffer = point.buffer(15000)  # 15 km in meters

            # Get the mean PM2.5 value within the buffer
            value = pm25_image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=buffer,
                scale=1000
            ).get('b1')

        return value.getInfo() if value else None
    except Exception:
        return None  # Handle missing values gracefully

#NO2_mol_per_m2

# Define the NO2 dataset


# Define a function to get NO2 value for a specific latitude and longitude
def get_no2_value(lat, lon):
    point = ee.Geometry.Point(lon, lat)

    # Filter the dataset for 2020 and get the mean value
    no2_image = no2_dataset.filterDate('2020-01-01', '2020-12-31').mean()

    try:
        # Get the NO2 value at the specified point
        value = no2_image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=1113.2  # Sentinel-5P resolution (1113.2 meters)
        ).get('tropospheric_NO2_column_number_density')  # The correct band name

        # If the value is None, check within a 15 km radius
        if value is None:
            # Create a buffer of 15 km around the point
            buffer = point.buffer(15000)  # 15 km in meters

            # Get the mean NO2 value within the buffer
            value = no2_image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=buffer,
                scale=1113.2
            ).get('tropospheric_NO2_column_number_density')

        return value.getInfo() if value else None
    except Exception:
        return None  # Handle cases where the value is not available

def extract_socio_urban_features(lat, lon, year, place_name):
    socio_urban_features = {}
    #common_dir_for_tif_files = "/content/drive/.shortcut-targets-by-id/17moBzZYxjt-iNNY6pL-mPx8nOm29CtVB/LULC_Datasets/Load_Dataset"

    # Apply the function to each row in the DataFrame
    socio_urban_features['degree_of_urbanization'] = get_degree_of_urbanization(lat, lon)
    socio_urban_features['night_light_intensity'] = get_night_light_intensity(lat, lon)
    socio_urban_features['population_density_per_km2'] = get_population_density(lat, lon)  #no of inhabitants of the cell
    socio_urban_features['built_area_km2'] = get_built_area_km2(lat, lon) #amount of square km of built-up surface in the cell
    socio_urban_features['built_height_m'] = get_built_height(lat, lon) #mount of built cubic meters per surface unit in the cell
    socio_urban_features['msz_class'] = extract_msz_value(lat, lon)
    socio_urban_features['gis_occurence'] = get_gisd30_value(lat, lon)
    socio_urban_features['NO2_mol_per_m2'] = get_no2_value(lat, lon)
    socio_urban_features['pm25(microg/10m3)'] = get_pm25_value(lat, lon)

    return socio_urban_features

