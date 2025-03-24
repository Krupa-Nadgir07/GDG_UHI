from .imports import *
from app.preprocess.sentinel import *
from app.preprocess.lulc import *
from app.preprocess.socio_urban import *
from app.model_codes.uhi_index import *
from app.model_codes.natural_manmade import *
from app.model_codes.month_pred import *

# Feature Extraction of all -- lulc, sentinel, socio-urban
def extract_features(place_name, lat, lon, year):
    features_dict = {}
    features_dict["place_name"] = place_name
    features_dict["latitude"] = lat
    features_dict["longitude"] = lon
    features_dict["year"] = year

    models_dir = f"G:/My Drive/5th_Sem_EL/LULC_Datasets/Exported_models"
    models_dir = f"{models_dir}/Sharpen"
    model_file = f"{models_dir}/model2_2000_256_sharpen_b16.pickle"
    sat_dir = f"G:/My Drive/Deployment_EarthEngineExports"
    # output_mask_dir = '/content/drive/MyDrive/Deployment_LULC_masks'
    output_mask_dir = f"G:/My Drive/Deployment_LULC_masks"

    satellite_features = extract_sentinel_features(lat, lon, year, place_name)
    if satellite_features is None:
        print("Insufficient bands for this location. Error extracting image.")
        return False
    features_dict.update(satellite_features)

    time.sleep(3)
    while not (os.path.exists(os.path.join(sat_dir, f"Satellite_{place_name}.tif"))):
        time.sleep(10)
    lulc_features = extract_lulc_features(place_name, sat_dir, model_file, output_mask_dir)
    features_dict.update(lulc_features)

    time.sleep(3)
    socio_urban_features = extract_socio_urban_features(lat, lon, year, place_name)
    features_dict.update(socio_urban_features)

    return features_dict

# Visualizing esch sentinel feature
def visualize_sentinel(features_tif, selected_index, color_map):
    index_to_band = {
        'ndvi': 2, 'nbr': 3, 'ndui': 4, 'gndvi': 5, 'ndwi': 6,
        'shade': 7, 'ndbi': 8, 'nmdi': 9, 'ndbsi': 10, 'nbadi': 11,
        'wvp': 12, 'aot': 13, 'lst': 14, 'savi': 15, 'uhsi': 16
    }

    band_number = index_to_band.get(selected_index)
    if band_number is None:
        raise ValueError(f"Invalid index selected: {selected_index}")
    
    with rasterio.open(features_tif) as src:
        band_data = src.read(band_number)

    band_min = np.min(band_data)
    band_max = np.max(band_data)
    norm = Normalize(vmin=band_min, vmax=band_max)
    normalized_data = norm(band_data) 
    print(band_min,band_max)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=normalized_data,  # The normalized data (2D array)
            colorscale=color_map,  # Apply the color map
            zmin=0,  # Normalized min value
            zmax=1   # Normalized max value
        )
    )

    fig.update_layout(
        title=f"{selected_index.upper()} Visualization",
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            scaleanchor="y",  # Lock aspect ratio between x and y axes
            constrain="domain"  # Prevent stretching beyond initial domain
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            constrain="domain"  # Prevent stretching
        ),
        autosize=False,
        width=600,
        height=600,
        dragmode="pan",  # Enable panning with fixed dimensions
    )
    sentinel_plot = fig.to_html(full_html=False)
    return sentinel_plot


# Displaying Satellite Image on sentinel features page
def display_sat(satellite_tif):
    with rasterio.open(satellite_tif) as src:
        # Read raster bands as RGB
        red = src.read(1)
        green = src.read(2)
        blue = src.read(3)
    
    # Normalize the RGB values to the range [0, 1]
    def normalize_band(band):
        return (band - band.min()) / (band.max() - band.min())

    red = normalize_band(red)
    green = normalize_band(green)
    blue = normalize_band(blue)

    # Stack the bands into an RGB image
    rgb_image = np.stack([red, green, blue], axis=-1)

    # plotly graph
    fig = go.Figure()
    fig.add_trace(go.Image(z=rgb_image))

    # pan_plugin = mpld3.plugins.BoxZoom
    fig.update_layout(
        # title="Satellite Image",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        autosize=True,
        showlegend=False,
        dragmode="zoom"  # Disable pan, only zoom is enabled
    )
    # Convert the plot to HTML and render
    satellite_plot = fig.to_html(full_html=False)

    return satellite_plot

# Convert .tif to .png
def tif_to_png(tiff_path):
    with tifffile.TiffFile(tiff_path) as tif:
        image = tif.asarray()
        from PIL import Image
        img = Image.fromarray(image)
        # img.save(output_path, "PNG")
        return img
    
import folium
from folium import plugins
import pandas as pd
import branca.colormap as cm


def add_markers_to_map(map_obj, df, label_column, col_name, colormap):
    for _, row in df.iterrows():
        latitude = row['latitude']
        longitude = row['longitude']
        label = row[label_column]
        intensity = row[col_name]
        color = colormap(intensity)  # Get color based on intensity value
        folium.CircleMarker(
            location=(latitude, longitude),
            radius=2,  # Size of the marker
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=f"{label} ({latitude}, {longitude}) - col_name: {intensity}"
        ).add_to(map_obj)

def add_legend(map_obj, colormap):
    colormap.caption = "Intensity Scale"
    map_obj.add_child(colormap)

def map_visualization(dataframe, col_name):
  # Create a colormap for Intensity

  min_intensity = dataframe[col_name].min()
  max_intensity = dataframe[col_name].max()
  # colormap = cm.LinearColormap(colors=['blue', 'green', 'yellow', 'red'], vmin=min_intensity, vmax=max_intensity)
  colormap = cm.LinearColormap(colors=['blue', 'green', 'yellow', 'orange', 'red', 'brown'], vmin=min_intensity, vmax=max_intensity)


  # Initialize the map centered at an approximate central location
  initial_latitude = dataframe['latitude'].mean()
  initial_longitude = dataframe['longitude'].mean()

  map_obj = folium.Map(location=[initial_latitude, initial_longitude], zoom_start=2.5, width="100%", height="600px")

  # Add markers to the map
  add_markers_to_map(map_obj, dataframe, label_column='place_name', col_name=col_name, colormap=colormap)

  # Add the legend
  add_legend(map_obj, colormap)

  # Save the map to an HTML file or display it inline
  # map_obj.save('locations_map.html')
  # Display the map if running in Jupyter Notebook
  return map_obj._repr_html_()

def plot_uhii(monthwise_data):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=monthwise_data, mode='lines+markers', name='UHII',
                             line=dict(color='teal', width=2), marker=dict(size=8)))

    fig.update_layout(
        title='UHII Month-wise Data',
        xaxis_title='Month',
        yaxis_title='UHII Value',
        template='plotly_dark', 
        showlegend=True,
        dragmode='zoom',  
        hovermode='closest',  
        autosize=True,
        height=400,
        width=800,  # Set height while width adapts dynamically
        margin=dict(l=20, r=20, t=50, b=50)  
    )
    return fig.to_html(full_html=False)