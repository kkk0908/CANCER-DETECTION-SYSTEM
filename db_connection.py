
from flask import Flask, request, render_template
import pymysql
from flask import flash
db = pymysql.connect("localhost", "root", "", "cancer_detection")

app = Flask(__name__)
#api = Api(app)
app.secret_key = 'supersecretkey'
app.config['SESSION_TYPE'] = 'filesystem'

@app.route('/')
def signup():
	return render_template('addExperts.html')	
'''def someName():
    cursor = db.cursor()
    sql = "SELECT * FROM employee"
    cursor.execute(sql)
    results = cursor.fetchall()
    return render_template('mysql_flask_html.html', results=results)'''

@app.route('/addExperts', methods = ['GET', 'POST'])
def addExperts(): 
	if request.method=='POST': 
		if not request.form['name'] or not request.form['email'] or not request.form['dept'] or not request.form['hosptl'] or not request.form['password']:
			flash('Please enter all the fields', 'error')
		else:
			name=request.form['name']
			email=request.form['email']
			pswd=request.form['password']
			deprt=request.form['dept'] 
			hosptl=request.form['hosptl']
			cursor=db.cursor()
			cursor.execute("insert into experts (name,email,paswd,deprt,hosptl) values (%s,%s,%s,%s,%s)",(name,email,pswd,deprt,hosptl))
			flash("record added successfully")
			return render_template('login.html')
	return render_template('addExperts.html')        
   

@app.route('/Login', methods = ['GET', 'POST'])
def Login(): 
	if request.method=='POST': 
		if not request.form['email'] or not request.form['password']:
			flash('Please enter all the fields', 'error')
		else:
			email=request.form['email']
			password=request.form['password']
			cursor=db.cursor()
			cursor.execute("SELECT * from experts where email=(%s) and paswd=(%s)",(email,password))
			results = cursor.fetchall()
			if len(results==1):
				session['email']=email
			return render_template('mysql_flask_html.html', results=results)
	return render_template('Login.html')
if __name__ == '__main__':
	app.run(debug=True)

