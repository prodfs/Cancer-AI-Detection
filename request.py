# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 17:22:05 2020

@author: hp
"""

import requests



url = 'http://localhost:5000/predict_api'

r = requests.post(url,json={'age':10, 'estimated_salary':1000})



print(r.json())