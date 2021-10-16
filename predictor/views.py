from django.shortcuts import render
from .apps import PredictorConfig
from django.http import JsonResponse, HttpResponse
from rest_framework.views import APIView
from .functions import sent_to_vect, enough_symptoms, chatbot_response, diagonsis_function, pred_model, le
import api.settings 
import joblib
import pickle
import json
from sklearn import preprocessing
import numpy as np
from keras.models import load_model
# Create your views here.

class call_model(APIView):
    def post(self,request):
        #if request.method == 'GET':
            # get query from request
        
        query = "djbsjkdsjkdhjkshdkhdjhjk"
        # print(request.body)
        # print(request)
        
        res = request.body.decode("utf-8")
        json_res = json.loads(res)
        result = json_res['message']

        print("h", result)
        # print(json.load(request.data))
        # print(request.data.bot_reply)
        # vectorize query

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
                    response = {'bot_reply': predicted_disease, 'possible_diseases': list_of_diseases}
                    return JsonResponse(response)
            else:
                return JsonResponse({'bot_reply': "Try something new."})

        else:
            print("1")
            query = result[-1]
            response = chatbot_response(query)
            print(response)
            return JsonResponse({'bot_reply': response})

        # query = "skin rash itching"
        
            #return render(request, 'base.html',{'bot_reply': predicted_disease})

        # else :
        #     response = chatbot_response(query)
        #     return JsonResponse({'bot_reply': response})