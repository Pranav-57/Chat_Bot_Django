# Importing Required Libraries
from sklearn import preprocessing
import joblib
import numpy as np
import re

from googletrans import Translator
from nltk.util import ngrams, pr
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

all_symptoms_list = (['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 
                      'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 
                      'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 
                      'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 
                      'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 
                      'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 
                      'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 
                      'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 
                      'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 
                      'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 
                      'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 
                      'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 
                      'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 
                      'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 
                      'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 
                      'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 
                      'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
                      'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 
                      'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 
                      'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 
                      'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 
                      'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 
                      'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze',
                      'prognosis'])

def sent_to_vect(sentence):
  #all_symptoms_list = (['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze', 'prognosis'])

  #Lower, clean and tokenize the sentence
  sentence = sentence.lower()
  sentence = re.sub(r'[^a-zA-Z0-9\s]', ' ', sentence)
  tokens = [token for token in sentence.split(" ") if token != ""]
 
  user_sym_list = []
  # 1 gram 
  for tok in tokens:
    user_sym_list.append(tok)

  # 2 gram
  if(len(tokens)>1):
    output2g = list(ngrams(tokens, 2))
    for tok in output2g:
      test = '_'.join(tok)
      user_sym_list.append(test)

  # 3 gram
  if(len(tokens) > 2):
    output3g = list(ngrams(tokens, 3))
    for tok in output3g:
      test = '_'.join(tok)
      user_sym_list.append(test)


  # 4 gram
  if(len(tokens) > 3):
    output4g = list(ngrams(tokens, 4))
    for tok in output4g:
      test = '_'.join(tok)
      user_sym_list.append(test)

  symp_vec = np.zeros(132) ################
  # Mark the symptoms found in the symp_vec
  for symp1 in user_sym_list:
    for i in range(0,133):
      if(all_symptoms_list[i] == symp1):
     # print(symp1," ", all_symptoms_list[i])
       symp_vec[i] = 1


  return symp_vec

def enough_symptoms(symp_vec):
  count = 0;
  for i in symp_vec:
    if(i== 1):
      count += 1
  if(count >= 1):
    return True
  return False

############################################################################################################
# using TF - IDF and Linear Kerner/ Cosing Similarity

# def Normalize(text):
#     remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
#     #word tokenization
#     word_token = nltk.word_tokenize(text.lower().translate(remove_punct_dict))
    
#     #remove ascii
#     new_words = []
#     for word in word_token:
#         new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
#         new_words.append(new_word)
    
#     #Remove tags
#     rmv = []
#     for w in new_words:
#         text=re.sub("&lt;/?.*?&gt;","&lt;&gt;",w)
#         rmv.append(text)
        
#     #pos tagging and lemmatization
#     tag_map = defaultdict(lambda : wn.NOUN)
#     tag_map['J'] = wn.ADJ
#     tag_map['V'] = wn.VERB
#     tag_map['R'] = wn.ADV
#     lmtzr = WordNetLemmatizer()
#     lemma_list = []
#     rmv = [i for i in rmv if i]
#     for token, tag in nltk.pos_tag(rmv):
#         lemma = lmtzr.lemmatize(token, tag_map[tag[0]])
#         lemma_list.append(lemma)
#     return lemma_list

# welcome_input = ("hello")
# # , "hi", "greetings", "sup", "what's up","hey",
# welcome_response = ["hi"]
# # , "hey", "how may I help you?", "hi there", "hello", "I am glad! You are talking to me"
# def welcome(user_response):
#     for word in user_response.split():
#         if word.lower() in welcome_input:
#             return random.choice(welcome_response)

# def generateResponse(user_response):
#     robo_response=''
#     sent_tokens.append(user_response)

#     TfidfVec = TfidfVectorizer(tokenizer=Normalize, stop_words='english')
#     tfidf = TfidfVec.fit_transform(sent_tokens)
#     #vals = cosine_similarity(tfidf[-1], tfidf)
#     vals = linear_kernel(tfidf[-1], tfidf)
#     idx=vals.argsort()[0][-2]

#     flat = vals.flatten()
#     flat.sort()
#     req_tfidf = flat[-2]
#     if(req_tfidf==0) or "tell me about" in user_response:
#         print("Checking Wikipedia")
#         if user_response:
#             #wk.set_lang('mr')
#             #robo_response = wk.summary(user_response)
#             return robo_response
#     else:
#         robo_response = robo_response+sent_tokens[idx]

#         return robo_response#wikipedia search

# def wikipedia_data(input):
#     reg_ex = re.search('tell me about (.*)', input)
#     try:
#         if reg_ex:
#             topic = reg_ex.group(1)
#             wiki = wk.summary(topic, sentences = 3)
#             return wiki
#     except Exception as e:
#             print("No content has been found")
###########################################################################################################
# DEEP LEARNING MODEL

import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import json
from keras.models import load_model
import random

lemmatizer = WordNetLemmatizer()

model = load_model(r'C:\Pranav Aher\Project\Django\Project\api\api\predictor\models\chatbot_model.h5')
intents = json.loads(open(r'C:\Pranav Aher\Project\Django\Project\api\api\predictor\models\IntentsDocument.json').read())
words = pickle.load(open(r'C:\Pranav Aher\Project\Django\Project\api\api\predictor\models\words.pkl','rb'))
classes = pickle.load(open(r'C:\Pranav Aher\Project\Django\Project\api\api\predictor\models\classes.pkl','rb'))
#Execute once
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

  #Execute once
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
  print(text)
  print("2")
  ints = predict_class(text, model)
  res = getResponse(ints, intents)
  return res

#############################################################################################################



pred_model = joblib.load(r'C:\Pranav Aher\Project\Django\Project\api\api\predictor\disease_esitmator.sav')
#pred_model = joblib.load('.\models\disease_esitmator.sav')
# Label Encoder = le # api\predictor\disease_esitmator.sav
le = preprocessing.LabelEncoder()
#le.classes_ = np.load('.\models\Lab_encoder.npy', allow_pickle=True) 
le.classes_ = np.load(r'C:\Pranav Aher\Project\Django\Project\api\api\predictor\Lab_encoder.npy', allow_pickle=True)

def diagonsis_function(sentence):
  symp_vec = sent_to_vect(sentence)
  probabs = pred_model.predict_proba(symp_vec.reshape(1,-1))
  n = 2
  indices = (-probabs).argsort()
  ind  = indices.tolist()
  list_prob_dis = []
  for i in range(41):
    if(probabs[0][ind[0][i]]>0.010):
      list_prob_dis.append({'disease' :le.inverse_transform([ind[0][i]])[0] ,'probablities':round(probabs[0][ind[0][i]]*100, 2)})
  return list_prob_dis


def translate_function(text):
  t = Translator()
  print(text)
  return t.translate(text, dest="hi")