from django.apps import AppConfig
from django.conf import settings
import os
import pickle
import joblib

class PredictorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'predictor'


# class PredictorConfig(AppConfig):

    # create path to models
    # path = os.path.join(settings.MODELS, 'disease_estimator.sav')
 
    # load models into separate variables
    # these will be accessible via this class
    # with open(path, 'rb') as pickled:
    #    data = pickle.load(pickled)
    # regressor = data['regressor']
    # # vectorizer = data['vectorizer']

    # pred_model = joblib.load('api\predictor\models\disease_esitmator.sav')