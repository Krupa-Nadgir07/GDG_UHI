import os
import ee

if not ee.data._credentials:
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    ee.Authenticate()
    ee.Initialize(project='lulc-444313')
    print('EE Initialized')
else:
    print('EE already initialized')

# Initializing your flask application
from app import app 

# Secret key to store session values
app.config['SECRET_KEY'] = os.urandom(24)  

# Run the flask application
if __name__ == '__main__':
    app.run(debug=True)
