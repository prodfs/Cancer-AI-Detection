import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
breast_cancer_detection = pickle.load(open('breast_cancer_detection.pkl', 'rb'))
prostate = pickle.load(open('prostate.pkl', 'rb'))
cervical_cancer_detection = pickle.load(open('cervial_cancer_detection.pkl', 'rb'))
lungs_cancer_detection = pickle.load(open('lung_cancer_detection.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/appointment')
def appointment():
    return render_template('appointment.html')
'''Breast cancer'''	
@app.route('/breast_cancer')
def breast_cancer():
		return render_template('breast_cancer.html')

@app.route('/predict_breast',methods=['POST'])
def predict_breast_cancer():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error', 'fractal dimension error',
       'worst radius', 'worst texture', 'worst perimeter', 'worst area',
       'worst smoothness', 'worst compactness', 'worst concavity',
       'worst concave points', 'worst symmetry', 'worst fractal dimension']
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = breast_cancer_detection.predict(df)
        
    if output == 0:
        res_val = "** breast cancer **"
    else:
        res_val = "no breast cancer"
        

    return render_template('breast_cancer.html', prediction_text='Patient has {}'.format(res_val))
	
''' PROSTATE CANCER '''	
@app.route('/prostate_cancer')
def prostate_cancer():
		return render_template('prostate_cancer.html')

	
@app.route('/predict_prostate',methods=['POST'])		
def predict_prostate_cancer():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['radius','texture','perimeter','area','smoothness','compactness','symmetry','fractal_dimension']
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = prostate.predict(df)
        
    if output == 0:
        res_val = "** prostate cancer **"
    else:
        res_val = "no prostate cancer"
		
    return render_template('prostate_cancer.html', prediction_text='Patient has {}'.format(res_val))

''' Cervical Cancer'''
@app.route('/cervical_cancer')
def cervical_cancer():
		return render_template('cervical_cancer.html')

@app.route('/predict_cervical',methods=['POST'])
def predict_cervical_cancer():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['Number of sexual partners','First sexual intercourse','Num of pregnancies','Smokes','Smokes (years)', 'Smokes (packs/year)',
'Hormonal Contraceptives','Hormonal Contraceptives (years)','IUD','IUD (years)','STDs','STDs (number)','STDs:condylomatosis',
'STDs:cervical condylomatosis','STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis','STDs:syphilis',
'STDs:pelvic inflammatory disease','STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV',                          
'STDs:Hepatitis B','STDs:HPV','STDs: Number of diagnosis','Dx:Cancer','Dx:CIN','Dx:HPV','Dx','Hinselmann','Schiller','Citology']                          
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = cervical_cancer_detection.predict(df)
        
    if output == 0:
        res_val = "** cervical cancer **"
    else:
        res_val = "no cervical cancer"
        
    return render_template('cervical_cancer.html', prediction_text='Patient has {}'.format(res_val))
''' Lungs Cancer'''
@app.route('/lungs_cancer')
def lungs_cancer():
		return render_template('lungs_cancer.html')

@app.route('/predict_lungs',methods=['POST'])
def predict_lungs_cancer():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['Smokes','AreaQ','Alkhol']                          
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = lungs_cancer_detection.predict(df)

    if output == 0:
        res_val = "** lungs cancer **"
    else:
        res_val = "no lungs cancer"
        
    return render_template('lungs_cancer.html', prediction_text='Patient has {}'.format(res_val))	
	
if __name__ == "__main__":
    app.run(debug=True)