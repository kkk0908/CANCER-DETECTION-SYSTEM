from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap 
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from scipy.sparse import hstack
import pickle
import os

app=Flask(__name__)
Bootstrap(app)
@app.route('/')
def index():
	return render_template('index.html')
@app.route('/predict',methods=['POST']) 
def predict():
	if request.method=='POST':
		gd=request.form['gene_data']
		vd=request.form['var_data']
		rep=request.form['report']
		g=pd.Series(gd)
		v=pd.Series(vd)
		st=pd.Series(rep)
		gene_vectorizer_pkl=open('gene_vec_pickle_file.pkl','rb')
		gene_vectorizer=pickle.load(gene_vectorizer_pkl)
		#load Varation pickle file
		var_vectorizer_pkl=open('var_vec_pickle_file.pkl','rb')
		variation_vectorizer=pickle.load(var_vectorizer_pkl)
		#load repor tpickle file
		text_vectorizer_pkl=open('report_vec_pickle_file.pkl','rb')
		text_vectorizer=pickle.load(text_vectorizer_pkl)
		#load naive bais classifier pickle file
		clf_pkl=open('naive_pickle_file.pkl','rb')
		clf=pickle.load(clf_pkl)
        #load probablity classifier!
		sig_clf_pkl=open('sig_clf_pickle_file.pkl','rb')
		sig_clf=pickle.load(sig_clf_pkl) 
		#one hot encoding
		test_gene_feature_onehotCoding = gene_vectorizer.transform(g)
		test_variation_feature_onehotCoding = variation_vectorizer.transform(v)
		test_text_feature_onehotCoding = text_vectorizer.transform(st)
		test_text_feature_onehotCoding = normalize(test_text_feature_onehotCoding, axis=0)
		test_gene_var_onehotCoding = hstack((test_gene_feature_onehotCoding,test_variation_feature_onehotCoding))
		test_x_onehotCoding = hstack((test_gene_var_onehotCoding, test_text_feature_onehotCoding)).tocsr()
		predicted_cls = sig_clf.predict(test_x_onehotCoding[0])

		
	prb=np.round(sig_clf.predict_proba(test_x_onehotCoding[0]),4).max()
	return render_template('CDSResult.html',prediction=predicted_cls[0],prob=prb)	

if __name__=='__main__':
	app.run(debug=True)	
