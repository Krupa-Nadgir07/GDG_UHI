from ..imports import *

models_dir = '"G:/My Drive/Natural_Manmade"'
all_features_names = ['place_name', 'Latitude', 'Longitude', 'year',
                     'NDVI_max', 'NDVI_mean', 'NDVI_min', 'NDVI_stdDev',
                     'NBR_max', 'NBR_mean', 'NBR_min', 'NBR_stdDev',
                     'NDUI_max', 'NDUI_mean', 'NDUI_min', 'NDUI_stdDev',
                     'GNDVI_max', 'GNDVI_mean', 'GNDVI_min', 'GNDVI_stdDev',
                     'NDWI_max', 'NDWI_mean', 'NDWI_min', 'NDWI_stdDev',
                     'Shade_max', 'Shade_mean', 'Shade_min', 'Shade_stdDev',
                     'NDBI_max', 'NDBI_mean', 'NDBI_min', 'NDBI_stdDev',
                     'NMDI_max', 'NMDI_mean', 'NMDI_min', 'NMDI_stdDev',
                     'NDBSI_max', 'NDBSI_mean', 'NDBSI_min', 'NDBSI_stdDev',
                     'NBaDI_max', 'NBaDI_mean', 'NBaDI_min', 'NBaDI_stdDev',
                     'WVP_max', 'WVP_mean', 'WVP_min', 'WVP_stdDev', 'AOT_max',
                     'AOT_mean', 'AOT_min', 'AOT_stdDev', 'LST_max', 'LST_mean',
                     'LST_min', 'LST_stdDev', 'SAVI_max', 'SAVI_mean', 'SAVI_min',
                     'SAVI_stdDev', 'UHSI_max', 'UHSI_mean', 'UHSI_min', 'UHSI_stdDev',
                     '2_ratio', '4_ratio', '5_ratio', '9_ratio', '1_ratio', '3_ratio', '6_ratio', '7_ratio', '8_ratio',
                     'degree_of_urbanization', 'night_light_intensity', 'population_density_per_km2', 'built_area_km2',
                     'built_height_m', 'msz_class', 'gis_occurence', 'NO2_mol_per_m2', 'pm25(microg/10m3)']

required_features_names = ['NDVI_max', 'NDVI_mean', 'NDVI_min', 'NDVI_stdDev', 'Shade_max',
       'Shade_mean', 'Shade_min', 'Shade_stdDev', 'NBR_max', 'NBR_mean',
       'NBR_min', 'NBR_stdDev', 'NDUI_max', 'NDUI_mean', 'NDUI_min',
       'NDUI_stdDev', 'GNDVI_max', 'GNDVI_mean', 'GNDVI_min', 'GNDVI_stdDev',
       'NDWI_max', 'NDWI_mean', 'NDWI_min', 'NDWI_stdDev', 'NDBI_max',
       'NDBI_mean', 'NDBI_min', 'NDBI_stdDev', 'NMDI_max', 'NMDI_mean',
       'NMDI_min', 'NMDI_stdDev', 'NDBSI_max', 'NDBSI_mean', 'NDBSI_min',
       'NDBSI_stdDev', 'NBaDI_max', 'NBaDI_mean', 'NBaDI_min', 'NBaDI_stdDev',
       'WVP_max', 'WVP_mean', 'WVP_min', 'WVP_stdDev', 'AOT_max', 'AOT_mean',
       'AOT_min', 'AOT_stdDev', 'LST_max', 'LST_mean', 'LST_min', 'LST_stdDev',
       'SAVI_max', 'SAVI_mean', 'SAVI_min', 'SAVI_stdDev', 'UHSI_max',
       'UHSI_mean', 'UHSI_min', 'UHSI_stdDev', '1_ratio', '2_ratio', '3_ratio',
       '4_ratio', '5_ratio', '6_ratio', '7_ratio', '8_ratio', '9_ratio',
       'degree_of_urbanization', 'night_light_intensity',
       'population_density_per_km2', 'built_up_surface',
       'avg_gross_built_height', 'morphological_settlement_zone',
       'global_impervious_surface_occurence', 'NO2_mol_per_m2',
       'pm25(microg/10m3)']


def extract_subset(input_dict, required_keys):
    subset = {}
    for key in required_keys:
        if key in input_dict:
            subset[key] = input_dict[key]
        return subset


def uhii_12_months(all_features, output_list_or_dict = True):
    models_dir = 'G:/My Drive/Natural_Manmade/'
    model_path = f'{models_dir}/uhii_stacking_model_2.pkl'
    scalar_path = f'{models_dir}/uhii_scaler.pkl'


    with open(scalar_path, 'rb') as file:
        loaded_scaler = pickle.load(file)

    with open(model_path, 'rb') as file:
        loaded_stacking_2_model = pickle.load(file)

    subset_features = extract_subset(all_features, required_features_names)

    for feature in required_features_names:
        if feature not in subset_features:
            subset_features[feature] = 0  # Use 0 or NaN or any appropriate value

    # Reorder the subset_features to match the trained model
    subset_features_reordered = {key: subset_features[key] for key in required_features_names}

    # Convert the reordered dictionary to a DataFrame
    X_test = pd.DataFrame([subset_features_reordered])

    # Prepare your new data
    X_test_scaled = loaded_scaler.transform(X_test)

    # Make predictions
    predictions = loaded_stacking_2_model.predict(X_test_scaled)

    # Plot the data
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.figure(figsize=(8, 5))
    plt.plot(months, predictions[0], marker='o', linestyle='-', color='green')
    plt.title('UHI Index (EA Method)', fontsize=12)
    plt.xlabel('Months', fontsize=12, color='purple')
    plt.ylabel('Intensity', fontsize=12, color='purple')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.show()

    prediction_dict = dict(zip(months, predictions[0]))

    if output_list_or_dict:
        return predictions[0]
    else:
        return prediction_dict