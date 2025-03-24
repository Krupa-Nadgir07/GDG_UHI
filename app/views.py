from flask import render_template, request, jsonify, redirect, url_for, session, send_file, render_template_string
from app import app
import time
from .utils import *
import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from io import BytesIO


dataset_dir = f"G:/My Drive/5th_Sem_EL/LULC_Datasets/Load_Dataset"
# df_all_cities_sat_socio = pd.read_csv(f"{dataset_dir}/all_cities_sat_socio.csv")
# df_all_villages_sat_socio = pd.read_csv(f"{dataset_dir}/all_villages_sat_socio.csv")
df_all_locations_sat_socio = pd.read_csv(f"{dataset_dir}/all_locations_sat_socio.csv")
# all_features = None

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/docs")
def docs():
    return render_template("doc.html")

# Globe Visualization for ROI
@app.route("/region_of_interest", methods=["GET", "POST"])
def roi():
    if request.method == 'POST' and request.is_json:   
        data = request.json

        name = data['display_name']
        latitude = data['lat']
        longitude = data['lon']

        session['address'] = name
        session['latitude'] = latitude
        session['longitude'] = longitude

        # print(data)
        # print(type(data))
        # return redirect("/dashboard") Redirection is done using javascript

    return render_template("globe_vis.html")


# Redirection page to all other pages. 
@app.route("/dashboard")
def dashboard():

    if 'address' not in session:
        return redirect('roi')

    place_name = session['address'].split(',')[0].strip()
    session['roi_lower'] = place_name.lower()
    session['country'] = session['address'].split(',')[-1].strip()
    city = place_name
    # print(session) 
    # if not all_features:
    place_name = place_name.replace(' ','_').lower()
    # if not all_features:

    # if 'all_features' not in session:
    all_features = extract_features(place_name, float(session['latitude']),float(session['longitude']),2023)
    print(all_features)
    session['all_features'] = json.loads(json.dumps(all_features, default=lambda x: x.item() if isinstance(x, np.generic) else x))
    
    features = [
        ('info', 0, 3),
        ('sentinel', 8, 67),
        ('ratios', 68, 76),
        ('socio_urban', 77, 85)
    ]

    keys = list(session['all_features'].keys())
    for category, start, end in features:
        session[category] = {keys[i]:session['all_features'][keys[i]] for i in range(start, end+1)}
    
    # del session['all_features']

    # print(session)

    # place_name = session['roi_lower']
    satellite_tif = f"G:/My Drive/Deployment_EarthEngineExports/Satellite_{session['roi_lower']}.tif"
    satellite_png = tif_to_png(satellite_tif)
    img_io = io.BytesIO()
    satellite_png.save(img_io, format='PNG')
    img_io.seek(0)  # Go to the start of the stream

    img = Image.open(img_io)
    width, height = img.size 
    area_covered = height*width/100
    # Encode the image to base64
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    return render_template("dashboard_page.html", info=session['info'], sat_img = img_base64, address=session['address'],area_covered=area_covered,city=city)



@app.route("/sentinel_features", methods=["GET", "POST"])
def sentinel():
    # Determine selected index and color map
    if request.method == "POST":
        selected_index = request.form.get("index", "ndvi")
        color_map = request.form.get("color", "viridis")
    else:
        selected_index = "ndvi"
        color_map = "viridis"

    # Retrieve session data
    place_name = session["roi_lower"]
    sentinel_params = session["sentinel"]
    features_tif = f"G:/My Drive/Deployment_EarthEngineExports/Features_{place_name}.tif"

    # Ensure selected index is uppercase
    selected_index_upper = selected_index.upper()

    if selected_index == 'shade':
        selected_index_upper = 'Shade'
    elif selected_index == 'nbadi':
        selected_index_upper = 'NBaDI'

    # Get min, max, std, and mean values safely
    mini = sentinel_params.get(f"{selected_index_upper}_min", 0)
    maxi = sentinel_params.get(f"{selected_index_upper}_max", 1)
    std = sentinel_params.get(f"{selected_index_upper}_stdDev", 0.1)
    mean = sentinel_params.get(f"{selected_index_upper}_mean", 0.5)

    try:
        sentinel_raster = visualize_sentinel(features_tif, selected_index, color_map)
    except ValueError as e:
        return render_template_string("<p>{{ error_message }}</p>", error_message=str(e))
    return render_template('sentinel.html',sentinel_raster=sentinel_raster, selected_index=selected_index_upper, color_map=color_map,info=session['info'],address=session['address'],\
                           mini=mini, maxi=maxi, std=std,mean=mean)
    # return render_template('sentinel.html',png_image_url=png_image_url )

@app.route("/socio_urban")
def socio_urban():
    socio_urban_params = session['socio_urban']
    return render_template("socio_urban.html", socio_urban_params=socio_urban_params)

@app.route("/analyse_uhi")
def uhi_main():
    city = session['address'].split(',')[0].strip()
    model_dir_prithivi = f"G:/My Drive/5th_Sem_EL/LULC_Datasets/UHI_Model_and_LLM_files"
    image_dir = f"G:/My Drive/Deployment_EarthEngineExports"
    all_features = session['all_features']

    ratio_labels = {
    '1_ratio': "Water",
    '2_ratio': "Trees",
    '3_ratio': "Flooded Vegetation",
    '4_ratio': "Crops",
    '5_ratio': "Built Area",
    '6_ratio': "Bare Ground",
    '7_ratio': "Snow/Ice",
    '8_ratio': "Clouds",
    '9_ratio': "Rangeland"
    }

    satellite_tif = f"G:/My Drive/Deployment_EarthEngineExports/Satellite_{session['roi_lower']}.tif"
    satellite_png = tif_to_png(satellite_tif)
    img_io = io.BytesIO()
    satellite_png.save(img_io, format='PNG')
    img_io.seek(0)  # Go to the start of the stream

    img = Image.open(img_io)
    width, height = img.size 
    area_covered = height*width/100
    # Encode the image to base64
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

    lulc = f"G:/My Drive/Deployment_LULC_masks/lulc_mask_{session['roi_lower']}.png"
    with open(lulc, "rb") as img_file:
        lulc_base64 = base64.b64encode(img_file.read()).decode('utf-8')

    ratios = session['ratios']
    sorted_ratios = sorted(ratios.items(), key=lambda item: item[1], reverse=True)[:3]
    mapped_top3 = [(ratio_labels.get(key.split('_')[0], key), value) for key, value in sorted_ratios]
    top1, top2, top3 = mapped_top3 if len(mapped_top3) >= 3 else (None, None, None)

    place_name = session['roi_lower'].replace(' ','_').lower()
    # uhi_predict = fetch_uhi(image_path=f"{image_dir}/Satellite_{place_name}.tif", model_dir_prithivi=model_dir_prithivi)
    natural_or_manmade = check_natural_or_manmade(all_features)
    monthwise = uhii_12_months(all_features,output_list_or_dict=True)
    monthwise_list = monthwise.tolist()
    uhi_predict = max(monthwise_list)
    plot_base64 = plot_uhii(monthwise_list)

    print(monthwise)
    return render_template("uhi_analysis.html", uhi_predict=uhi_predict, natural_or_manmade=natural_or_manmade,sorted_ratios=sorted_ratios,\
                            top1=top1,sat_img = img_base64, address=session['address'],area_covered=area_covered,
                            top2=top2,lulc_img = lulc_base64,city=city,
                            top3=top3,info=session['info'],plot_base64=plot_base64)

# @app.route("/mitigation")
# def llm():
#     return render_template('llm.html')

@app.route("/world_wide_analysis", methods=["GET", "POST"])
def world():
    map_html = map_visualization(df_all_locations_sat_socio, 'NDVI_mean')

    if request.method == 'POST':
        sentinel_feature = request.form.get('sentinel_feature')
        statistical_feature = request.form.get('statistical_feature')
        socio_urban_feature = request.form.get('socio_urban_feature')

        if sentinel_feature:
            map_html = map_visualization(df_all_locations_sat_socio, f'{sentinel_feature}_{statistical_feature}')
        
        elif socio_urban_feature:
            map_html = map_visualization(df_all_locations_sat_socio, f'{socio_urban_feature}')
        
        # return redirect('world_wide_analysis')

    return render_template('world.html',map_html=map_html)
    