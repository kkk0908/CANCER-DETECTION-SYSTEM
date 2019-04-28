from flask import Flask, request, flash, url_for, redirect, render_template
from flask_sqlalchemy import SQLAlchemy
#from form import RegistrationForm , LoginForm
from datetime import datetime
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Cancer_experts.db'
app.config['SECRET_KEY'] = "AnyRandomString"

db = SQLAlchemy(app)

class doctors(db.Model):  #this class is the table inside the cancer_experts database 
   id = db.Column(db.Integer, primary_key = True)
   name = db.Column(db.String(30),unique=True, nullable=False)
   email = db.Column(db.String(50), unique=True, nullable=False )  
   department = db.Column(db.String(200), nullable=False, default='Cancer_Specialist')
   hospital_name = db.Column(db.String(200), nullable=False)
   password = db.Column(db.String(30),unique=True, nullable=False)
   query=db.relationship('Search',backref='patient_name',lazy=True)
   def __repr__(self):#how object isbeing printed
          return f"doctors('{self.name}','{self.email}','{self.department}','{self.hospital_name}','{self.password}')"




class Search(db.Model):
	id=db.Column(db.Integer, primary_key = True)
	gene = db.Column(db.String(30))
	var = db.Column(db.String(30))
	date = db.Column(db.DateTime,nullable=False,default=datetime.utcnow)
	doct_id=db.Column(db.Integer,db.ForeignKey('doctors.id'),nullable=False)
	def __repr__(self):
		return f"Search('{self.gene}','{self.var}','{self.date}')"
'''def __init__(self, name, email, department,hospital_name):
   self.name = name
   self.email = email
   self.department = department
   self.hospital_name = hospital_name'''


@app.route('/')
def show_all():
   return render_template('show_all.html', doctors = Search.query.all() )
@app.route('/new', methods = ['GET', 'POST'])
def new():
   if request.method == 'POST':
      if not request.form['name'] or not request.form['email'] or not request.form['dept'] or not request.form['hosptl']:
         flash('Please enter all the fields', 'error')
      else:
         student = students(request.form['name'], request.form['email'],
            request.form['dept'], request.form['hosptl'])
         
         db.session.add(student)
         db.session.commit()
         flash('Record was successfully added')
         return redirect(url_for('show_all'))
   return render_template('new.html')

if __name__ == '__main__':
   db.create_all()
   app.run(debug = True)   