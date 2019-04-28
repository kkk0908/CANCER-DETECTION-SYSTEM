from flask import Flask,render_template,url_for,request,session,redirect
from flask_bootstrap import Bootstrap 
from flask import flash
import os
import pymysql
import pandas as pd
import re
import numpy as np
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.classification import accuracy_score, log_loss
from scipy.sparse import hstack
from collections import Counter, defaultdict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import math
from sklearn.metrics import normalized_mutual_info_score
from collections import defaultdict
from sklearn.externals import joblib


PEOPLE_FOLDER = os.path.join('static', 'image')
db = pymysql.connect("localhost", "root", "", "cancer_detection")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
app.config['SESSION_TYPE'] = 'filesystem'
app.secret_key = 'supersecretkey'
Bootstrap(app)
@app.route('/')
def index():
	'''cursor = db.cursor()
	sql = "SELECT * FROM experts"
	cursor.execute(sql)
	results = cursor.fetchall()
	return render_template('mysql_flask_html.html', results=results)'''

	#if session.get('logged_in') = True:
	img1 = os.path.join(app.config['UPLOAD_FOLDER'], 'img3.jpg')
	img2=os.path.join(app.config['UPLOAD_FOLDER'],'img1.jpg')
	img3 = os.path.join(app.config['UPLOAD_FOLDER'], 'img1.jpg')
	img4 = os.path.join(app.config['UPLOAD_FOLDER'], 'img4.jpg')
	img5 = os.path.join(app.config['UPLOAD_FOLDER'], 'img2.jpg')
	return render_template('index.html',image1 = img1,image2=img2,image3=img3,image4=img4,image5=img5)
		

def extract_dictionary_paddle(cls_text):
	    dictionary = defaultdict(int)
	    for index, row in cls_text.iterrows():
	        for word in row['TEXT'].split():
	            dictionary[word] +=1
	    return dictionary
@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect('/')
@app.route('/signup')
def signup():
	return render_template('registration.html')    
@app.route('/registration', methods = ['GET', 'POST'])
def registration(): 
	if request.method=='POST': 
		if not request.form['fname'] or not request.form['email'] or not request.form['dept'] or not request.form['hosptl'] or not request.form['password']:
			flash('Please enter all the fields', 'error')
		else:
			fname=request.form['fname']
			lname=request.form['lastname']
			name=fname+lname
			email=request.form['email']
			pswd=request.form['password']
			deprt=request.form['dept'] 
			hosptl=request.form['hosptl']
			print(name,email,pswd,deprt,hosptl)
			cursor=db.cursor()
			cursor.execute("INSERT INTO experts(name,email,paswd,deprt,hosptl) values (%s,%s,%s,%s,%s)",(name,email,pswd,deprt,hosptl))
			db.commit()
			flash("record added successfully")
			return redirect('/')
	return render_template('registration.html')
@app.route('/login', methods=['GET', 'POST'])
def login():
    error=None
    if request.method == 'POST':
        email  = request.form['email']
        password_cand  = request.form['password']
        cur = db.cursor()
        result=0
        cur.execute("SELECT * FROM experts WHERE email= %s",[email])
        for i in cur:
            if i==None:
                result=0
                break
            else:
                result=1
                break
        if result > 0:
            cur.execute("SELECT paswd FROM experts WHERE email =%s",[email])
            
            data = cur.fetchone()
            password = data[0]
            
            if password_cand==password:
                session['logged_in']= True
                session['email'] = email
                return render_template('home.html',result=session['email'])
                flash('You are now logged in')

            else:
                flash('Invalid login')
                return redirect('/')
                
            cur.close()
        else:
            flash('username not found')
            return redirect('/')
    return redirect('/')

@app.route('/hpme')
def home():
	img1 = os.path.join(app.config['UPLOAD_FOLDER'], 'img3.jpg')
	img2=os.path.join(app.config['UPLOAD_FOLDER'],'img1.jpg')
	img3 = os.path.join(app.config['UPLOAD_FOLDER'], 'img1.jpg')
	img4 = os.path.join(app.config['UPLOAD_FOLDER'], 'img4.jpg')
	img5 = os.path.join(app.config['UPLOAD_FOLDER'], 'img2.jpg')
	return render_template('home.html',image1 = img1,image2=img2,image3=img3,image4=img4,image5=img5,result=session['email'])

@app.route('/predict',methods=['POST']) 
def predict():
	data = pd.read_csv('training_variants')
	data_text =pd.read_csv("training_text",sep="\|\|",engine="python",names=["ID","TEXT"],skiprows=1)
	stop_words = set(stopwords.words('english'))
	for index, row in data_text.iterrows():
		total_text=row['TEXT']
		column='TEXT'
		if type(total_text) is not int:
			string = ""
			total_text = re.sub('[^a-zA-Z0-9\n]', ' ', str(total_text))
			total_text = re.sub('\s+',' ', total_text)
			total_text = total_text.lower()
			for word in total_text.split():
				if not word in stop_words:
					string += word + " "
			data_text[column][index]=string

	result = pd.merge(data, data_text,on='ID', how='left')

	y_true = result['Class'].values
	result.Gene      = result.Gene.str.replace('\s+', '_')
	result.Variation = result.Variation.str.replace('\s+', '_')

	# split the data into test and train by maintaining same distribution of output varaible 'y_true' [stratify=y_true]
	X_train, test_df, y_train, y_test = train_test_split(result, y_true, stratify=y_true, test_size=0.2)
	# split the train data into train and cross validation by maintaining same distribution of output varaible 'y_train' [stratify=y_train]
	train_df, cv_df, y_train, y_cv = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2)



	# one-hot encoding of Gene feature.
	gene_vectorizer = CountVectorizer()
	train_gene_feature_onehotCoding = gene_vectorizer.fit_transform(train_df['Gene'])
	test_gene_feature_onehotCoding = gene_vectorizer.transform(test_df['Gene'])
	cv_gene_feature_onehotCoding = gene_vectorizer.transform(cv_df['Gene'])

	# one-hot encoding of variation feature.
	variation_vectorizer = CountVectorizer()
	train_variation_feature_onehotCoding = variation_vectorizer.fit_transform(train_df['Variation'])
	test_variation_feature_onehotCoding = variation_vectorizer.transform(test_df['Variation'])
	cv_variation_feature_onehotCoding = variation_vectorizer.transform(cv_df['Variation'])


	
	
	# building a CountVectorizer with all the words that occured minimum 3 times in train data
	text_vectorizer = CountVectorizer(min_df=3)
	train_text_feature_onehotCoding = text_vectorizer.fit_transform(train_df['TEXT'])
	# getting all the feature names (words)
	train_text_features= text_vectorizer.get_feature_names()

	# train_text_feature_onehotCoding.sum(axis=0).A1 will sum every row and returns (1*number of features) vector
	train_text_fea_counts = train_text_feature_onehotCoding.sum(axis=0).A1

	# zip(list(text_features),text_fea_counts) will zip a word with its number of times it occured
	text_fea_dict = dict(zip(list(train_text_features),train_text_fea_counts))



	dict_list = []
	for i in range(1,10):
	    cls_text = train_df[train_df['Class']==i]
	    # build a word dict based on the words in that class
	    dict_list.append(extract_dictionary_paddle(cls_text))
	    # append it to dict_list

	total_dict = extract_dictionary_paddle(train_df)


	confuse_array = []
	for i in train_text_features:
	    ratios = []
	    max_val = -1
	    for j in range(0,9):
	        ratios.append((dict_list[j][i]+10 )/(total_dict[i]+90))
	    confuse_array.append(ratios)
	confuse_array = np.array(confuse_array)

	# don't forget to normalize every feature
	train_text_feature_onehotCoding = normalize(train_text_feature_onehotCoding, axis=0)

	# we use the same vectorizer that was trained on train data
	test_text_feature_onehotCoding = text_vectorizer.transform(test_df['TEXT'])
	# don't forget to normalize every feature
	test_text_feature_onehotCoding = normalize(test_text_feature_onehotCoding, axis=0)

	# we use the same vectorizer that was trained on train data
	cv_text_feature_onehotCoding = text_vectorizer.transform(cv_df['TEXT'])
	# don't forget to normalize every feature
	cv_text_feature_onehotCoding = normalize(cv_text_feature_onehotCoding, axis=0)

	sorted_text_fea_dict = dict(sorted(text_fea_dict.items(), key=lambda x: x[1] , reverse=True))
	sorted_text_occur = np.array(list(sorted_text_fea_dict.values()))


	train_gene_var_onehotCoding = hstack((train_gene_feature_onehotCoding,train_variation_feature_onehotCoding))
	test_gene_var_onehotCoding = hstack((test_gene_feature_onehotCoding,test_variation_feature_onehotCoding))
	cv_gene_var_onehotCoding = hstack((cv_gene_feature_onehotCoding,cv_variation_feature_onehotCoding))

	train_x_onehotCoding = hstack((train_gene_var_onehotCoding, train_text_feature_onehotCoding)).tocsr()
	train_y = np.array(list(train_df['Class']))

	test_x_onehotCoding = hstack((test_gene_var_onehotCoding, test_text_feature_onehotCoding)).tocsr()
	test_y = np.array(list(test_df['Class']))

	cv_x_onehotCoding = hstack((cv_gene_var_onehotCoding, cv_text_feature_onehotCoding)).tocsr()
	cv_y = np.array(list(cv_df['Class']))




	

	alpha = [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100,1000]
	cv_log_error_array = []
	for i in alpha:
	    #print("for alpha =", i)
	    clf = MultinomialNB(alpha=i)
	    clf.fit(train_x_onehotCoding, train_y)
	    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
	    sig_clf.fit(train_x_onehotCoding, train_y)
	    sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)
	    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))
	    # to avoid rounding error while multiplying probabilites we use log-probability estimates
	    #print("Log Loss :",log_loss(cv_y, sig_clf_probs)) 

	best_alpha = np.argmin(cv_log_error_array)
	clf = MultinomialNB(alpha=alpha[best_alpha])
	clf.fit(train_x_onehotCoding, train_y)
	sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
	sig_clf.fit(train_x_onehotCoding, train_y)
	sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)

	if request.method=='POST':
		gd=request.form['gene_data']
		vd=request.form['var_data']
		rep=request.form['report']
		g=pd.Series(gd)
		v=pd.Series(vd)
		st=pd.Series(rep)
	
		test_gene_feature_onehotCoding = gene_vectorizer.transform(g)
		test_variation_feature_onehotCoding = variation_vectorizer.transform(v)
		test_text_feature_onehotCoding = text_vectorizer.transform(st)
		test_text_feature_onehotCoding = normalize(test_text_feature_onehotCoding, axis=0)
		test_gene_var_onehotCoding = hstack((test_gene_feature_onehotCoding,test_variation_feature_onehotCoding))
		test_x_onehotCoding = hstack((test_gene_var_onehotCoding, test_text_feature_onehotCoding)).tocsr()

		no_feature = 100
		predicted_cls = sig_clf.predict(test_x_onehotCoding[0])
		#print("Predicted Class :", predicted_cls[0])
		#print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[0]),4))
	prb=np.round(sig_clf.predict_proba(test_x_onehotCoding[0]),4).max()
	prb*=100
	if prb<=33:
		prb=prb*3
	elif prb>33 and prb<50:
	    prb*=2
	elif prb>50 and prb<=65:
	    prb=prb*1.5     	

	return render_template('CDSResult.html',prediction=predicted_cls[0],prob=prb)	


if __name__=='__main__':
	app.secret_key='sksksksksksk'
	app.run(debug=True)


