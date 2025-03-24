from ..imports import *

models_dir = 'G:/My Drive/Natural_Manmade'
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

required_features_names = ['Latitude', 'Longitude', 'degree_of_urbanization',
       'night_light_intensity', 'population_density_per_km2',
       'built_up_surface', 'avg_gross_built_height',
       'morphological_settlement_zone', 'global_impervious_surface_occurence',
       'NO2_mol_per_m2', 'pm25(microg/10m3)', 'NDVI_max', 'NDVI_mean',
       'NDVI_min', 'NDVI_stdDev', 'Shade_max', 'Shade_mean', 'Shade_min',
       'Shade_stdDev', 'NBR_max', 'NBR_mean', 'NBR_min', 'NBR_stdDev',
       'NDUI_max', 'NDUI_mean', 'NDUI_min', 'NDUI_stdDev', 'GNDVI_max',
       'GNDVI_mean', 'GNDVI_min', 'GNDVI_stdDev', 'NDWI_max', 'NDWI_mean',
       'NDWI_min', 'NDWI_stdDev', 'NDBI_max', 'NDBI_mean', 'NDBI_min',
       'NDBI_stdDev', 'NMDI_max', 'NMDI_mean', 'NMDI_min', 'NMDI_stdDev',
       'NDBSI_max', 'NDBSI_mean', 'NDBSI_min', 'NDBSI_stdDev', 'NBaDI_max',
       'NBaDI_mean', 'NBaDI_min', 'NBaDI_stdDev', 'WVP_max', 'WVP_mean',
       'WVP_min', 'WVP_stdDev', 'AOT_max', 'AOT_mean', 'AOT_min', 'AOT_stdDev',
       'LST_max', 'LST_mean', 'LST_min', 'LST_stdDev', 'SAVI_max', 'SAVI_mean',
       'SAVI_min', 'SAVI_stdDev', 'UHSI_max', 'UHSI_mean', 'UHSI_min',
       'UHSI_stdDev', '1_ratio', '2_ratio', '3_ratio', '4_ratio', '5_ratio',
       '6_ratio', '7_ratio', '8_ratio', '9_ratio']


def extract_subset(input_dict, required_keys):
  subset = {}
  for key in required_keys:
    if key in input_dict:
      subset[key] = input_dict[key]
  return subset


def check_natural_or_manmade(all_features):

  # Load the trained RandomForest model from the saved file
  model_path = 'G:/My Drive/Natural_Manmade/LogReg_classifier-all.pkl'
  model = joblib.load(model_path)

  # Get the model's expected feature names (excluding the target 'Label')
  required_features_names = model.feature_names_in_
  subset_features = extract_subset(all_features, required_features_names)

  # Remove 'Label' if it appears in the list of required features
  required_features_names = [feature for feature in required_features_names if feature != 'Label']

  # Ensure all required features are in the data (add placeholders for missing features)
  for feature in required_features_names:
      if feature not in subset_features:
          subset_features[feature] = 0  # Use 0 or NaN or any appropriate value

  # Reorder the subset_features to match the trained model
  subset_features_reordered = {key: subset_features[key] for key in required_features_names}

  # Convert the reordered dictionary to a DataFrame
  X_test = pd.DataFrame([subset_features_reordered])

  # Make predictions on the test data
  predictions = model.predict(X_test)

  # Output predictions
  print(f"Predictions: {predictions}")
  return "Man-made" if predictions[0] == 1 else "Natural"