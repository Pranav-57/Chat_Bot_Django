from django.shortcuts import render
from .apps import PredictorConfig
from django.http import JsonResponse, HttpResponse
from rest_framework.views import APIView
from .functions import sent_to_vect, enough_symptoms, chatbot_response, diagonsis_function, pred_model, le, translate_function
import api.settings 
import joblib
import pickle
import json
from sklearn import preprocessing
import numpy as np
from keras.models import load_model
from deep_translator import GoogleTranslator
from django.views.decorators.csrf import csrf_exempt

# Create your views here.

class call_model(APIView):
    def post(self,request):        
        query = "djbsjkdsjkdhjkshdkhdjhjk"        
        res = request.body.decode("utf-8")
        json_res = json.loads(res)
        messages = json_res['message']
        code = json_res['code']
        print('messages', messages)
        if not code == "en":
            result = []
            for message in messages:
                translated = GoogleTranslator(source='auto', target="en").translate(message)
                result.append(translated)
        
        else:
            result = messages

        print("result", result)

        if json_res['isSymptoms']:
            if enough_symptoms(sent_to_vect(result[-1])):
                query = ""
                for i in result:
                    query += i
                    query += " "

                vector = sent_to_vect(query)
                if(enough_symptoms(vector)):
                    vector = vector.reshape(1,-1)
                    prediction = pred_model.predict(vector)
                    predicted_disease = le.inverse_transform(prediction)[0]
                    list_of_diseases =  diagonsis_function(query)
                    
                    output = ""
                    # output = translator.translate(predicted_disease)
                    # output = t.translate(predicted_disease, dest=code)
                    translate_langauge_list = []
                    
                    for disease in list_of_diseases:
                        disease_output = GoogleTranslator(source='auto', target=code).translate(disease['disease']) 
                        
                        my_dic = {
                            'disease' : disease_output,
                            "probablities" : disease['probablities']
                        }
                        translate_langauge_list.append(my_dic)

                    response = {'bot_reply': output, 'possible_diseases': translate_langauge_list}
                    return JsonResponse(response)
            else:
                output = GoogleTranslator(source='auto', target=code).translate("Any other symptoms ?") 
                return JsonResponse({'bot_reply': output})

        else:
            query = result[-1]
            print(query)
            response = chatbot_response(query)
            output = GoogleTranslator(source='auto', target=code).translate(response) 
            # output = t.translate(response, dest="hi")
            print(output)
            return JsonResponse({'bot_reply': output})

@csrf_exempt
def translate_langauge(request):
    res = request.body.decode("utf-8")
    json_res = json.loads(res)
    message = json_res['message']
    code = json_res['code']
    
    translated = GoogleTranslator(source='auto', target=code).translate(message) 
    
    return JsonResponse({"translation" : translated})