from flask import Flask, render_template, session, redirect, url_for, flash, request, Response
from flask_bootstrap import Bootstrap
from flask_moment import Moment
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
# from test import tt
from sklearn.datasets import load_digits
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import json


app = Flask(__name__,static_url_path='')
app.config['SECRET_KEY'] = 'hard to guess string'

bootstrap = Bootstrap(app)
moment = Moment(app)
global transformer_batch
transformer_batch = IncrementalPCA(n_components=8, batch_size=200)

# class IndexForm(FlaskForm):
#     train_signal = SubmitField('train')
    # name = StringField('Please input the train time', validators=[DataRequired()])
    # submit = SubmitField('Submit')
    # algorithm = StringField('Please input the train method', validators=[DataRequired()])
    # algorithm_submit = SubmitField('Choose')
    # train_signal = SubmitField('train')
    # train = StringField('Please input whether training', validators=[DataRequired()])
    # train_submit = SubmitField('Train')

class ConfigForm(FlaskForm):
    train_time = StringField('Please input the train time', validators=[DataRequired()])
    train_time_submit = SubmitField('Submit')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


@app.route('/', methods=['GET', 'POST'])
def index():
    # _Index_form = IndexForm(request.form)
    # if form.validate_on_submit():
    #     old_name = session.get('name')
    #     old_algorithm = session.get('algorithm')
    #     session['algorithm']=request.form.get('algorithm')
    #     session['name'] = form.name.data
    # _threshold = session.get('threshold')
    # _threshold_front = float('%.2f' % _threshold)
    return render_template('index.html',train_time='1000 By Default',score='100',health_condition='Good')

@app.route('/Main/<train_flag>', methods=['GET', 'POST'])
def Main(train_flag):
	print('=============train_flag is',train_flag,'===========')
	if train_flag == '1':
	    print('================ Prepare =================')
	    session['train_iteration'] = 10000
	    session['progress']=0
	    session['left']=10000
	    return redirect(url_for('ipca',progress=0))
	else:
		_threshold = session.get('threshold')
		_threshold = float(_threshold)
		_threshold_front = float('%.4f' % _threshold)
		_current = session.get('current')
		_current = float(_current)
		_current_front = float('%.4f' % _current)
		return render_template('index.html',train_time=session.get('train_time'),threshold=_threshold_front,current=_current_front,score=session.get('score'),health_condition=session.get('health_condition'))
		# return render_template('Configuration.html',form=_Config_form,train_time=session.get('train_time'))
@app.route('/ipca/<progress>',methods=['GET','POST'])
def ipca(progress):
    # print("===================Begin===============")
    if int(progress) == 10000:
        print('=========training finished!========')
        session['threshold']=transformer_batch.explained_variance_ratio_.sum()
        session['progress']=0
        session['left']=10000
        return redirect(url_for('Main',train_flag=0))
    else:
        # train_iteration=1000
        # train_sample=30000
        # train_component=12 #this number need to be even for the convenience of testing
        # X, _ = load_digits(return_X_y=True)
        # X=np.random.rand(train_sample,train_component)
        # global transformer_batch
        #transformer_batch = IncrementalPCA(n_components=8, batch_size=200)
        # either partially fit on smaller batches of data
        X = np.random.rand(300000,12)
        for X_batch in np.array_split(X, 1000):
            transformer_batch.partial_fit(X_batch)
        # count+=1
        progress= int(progress) + 1000
        session['progress'] = progress
        print('---',session.get('progress'))
        session['left'] = 10000 - progress
        # print('=========training finished!========')
        # session['threshold']=transformer_batch.explained_variance_ratio_.sum()
        return redirect(url_for('ipca',progress=progress))

@app.route('/progress',methods=['GET','POST'])
def progress():
    jsonData = {}
    # print(session.get('progress'))
    _progress = session.get('progress')
    _left = session.get('left')
    jsonData['progress'] = _progress
    jsonData['left'] = _left
    print(_progress,_left)
    j=json.dumps(jsonData)
    print(j)
    return j

@app.route('/test',methods=['GET','POST'])
def test():
    # train_iteration=1000
    # train_sample=30000
    # train_component=12 #this number need to be even for the convenience of testing
    # pca = PCA(n_components = 8)

    # # X, _ = load_digits(return_X_y=True)
    # X=100*np.random.rand(train_sample,train_component)
    # print(X,X.shape)
    # # transformer_batch = IncrementalPCA(n_components=8, batch_size=200)
    # # either partially fit on smaller batches of data


    # for X_batch in np.array_split(X, train_iteration):
    #     print(X_batch.shape)
    #     transformer_batch.partial_fit(X_batch)
    X_test = np.random.rand(100,12)
    transformer_batch.partial_fit(X_test)
    session['current']=transformer_batch.explained_variance_ratio_.sum()
    if session['current']  - 0.05<= session.get('threshold'):
        session['score'] = np.random.randint(82,95)
        session['health_condition'] = 'Good'
    else:
        session['score'] = np.random.randint(40,65)
        session['health_condition'] = 'Bad'
    return redirect(url_for('Main',train_flag=0))

@app.route('/Configuration',methods=['GET','POST'])
def Configuration():
    # print('=============train_flag is',train_flag,'===========')
    # if train_flag == '1':
    #     print('================ Prepare =================')
    #     session['train_iteration'] = 10000
    #     session['progress']=0
    #     session['left']=10000
    #     return redirect(url_for('ipca',progress=0))
    # else:
    _Config_form = ConfigForm(request.form)
    # _Index_form = IndexForm(request.form)
    # # return redirect(url_for('index'))
    if request.method == 'POST':
        _train_time_store = request.form.get('train_time_input')
        session['train_time'] = _train_time_store
        return redirect(url_for('Configuration'))
    print('-----999------')
    # if _Config_form.validate_on_submit():
    #     _train_time=_Config_form.train_time.data
    #     return render_template('index.html',form=_Index_form,train_time=_train_time)
    return render_template('Configuration.html',form=_Config_form,train_time=session.get('train_time'))


if __name__=='__main__':
    app.run(port=7000,debug=True,threaded=True)