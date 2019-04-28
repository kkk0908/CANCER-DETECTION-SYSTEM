from flask import Flask, request, render_template
import pymysql
#from mytvpy.models import base

db = pymysql.connect("localhost", "root", "", "test")

app = Flask(__name__)
api = Api(app)

@app.route('/')
def someName():
    cursor = db.cursor()
    sql = "SELECT * FROM employee"
    cursor.execute(sql)
    results = cursor.fetchall()
    return render_template('mysql_flask_html.html', results=results)

if __name__ == '__main__':
app.run(debug=True)'''