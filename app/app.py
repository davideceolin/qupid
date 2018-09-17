#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import pandas as pd
import functools
from joblib import Parallel, delayed
import multiprocessing
from flask import Flask, send_from_directory
from multiprocessing import Process
from goose3 import Goose
from textblob import TextBlob
from textatistic import Textatistic
import urllib.request
import re
import os
import time
import glob
import pandas as pd
import requests
from urllib.parse import urlsplit
from twitterscraper import query_tweets
from gensim.summarization import keywords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import pickle
import numpy
from gensim import corpora, models
import gensim
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import os.path
import time
import json
from flask import Flask, render_template, request, redirect, session, flash, url_for, send_file, request, render_template
import os
import unicodedata
import time
import urllib
from MagicGoogle import MagicGoogle
from multiprocessing import Pool, cpu_count
from queue import Queue
from threading import Thread
import time
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
from flask import Flask, send_from_directory, send_file, Markup
from multiprocessing import Process
import matplotlib; matplotlib.use('Agg');
import matplotlib.pyplot as plt,mpld3
from markupsafe import Markup

from matplotlib import colors as mcolors


app = Flask(__name__)
app.secret_key = 'ThisIsSecret'
if __name__ == '__main__':
    app.run(debug=True)

def conv(s):
    try:
       return int(s)
    except ValueError:
       return s

def most_common(lst):
    return max(set(lst), key=lst.count)



def countLetters(word):
    count = 0
    for c in word:
        count += 1
    return count




def f1(q, url):
    # try:
    #print("Start: %s" % time.ctime())
    # Instead of returning the result we put it in shared queue.
    #     st = "/&callback=process&key=57bf606e01a24537ac906a86dc27891f94a0f587"
    #     # zz = urlopen ( url )
    #     quez = 'http://api.mywot.com/0.4/public_link_json2?hosts=' + url + st
    #     stt = urllib.request.urlopen(quez).read()
    #     stt = str(stt)
    #     wot = re.findall('\d+', stt)
    #     ##z=[[conv(s) for s in line.split()] for line in wot]
    #     z = [conv(s) for s in wot]
    #     high = (z[1])
    #     low = (z[2])
    #     # print ( high , low )
        # WAYBACK
        #zz = "{0.scheme}://{0.netloc}/".format(urlsplit(url))
        #zurlz = "https://web.archive.org/web/0/" + str(zz)
        #r = requests.get(zurlz, allow_redirects=True)
        #data = r.content
        #years = re.findall('\d+', str(data))
        #years = [conv(s) for s in years]
        #years = (years[0])
        #years = int(str(years)[:4])
        years = 1998
        cols = {'yeararchive': [years],
                # 'lowwot': [low],
                # 'highwot': [high],
                #'reponsetime': [vals],
                'url': [str(url)]}
        dfb = pd.DataFrame.from_dict(cols)
        #print(dfb)
        #print("Start: %s" % time.ctime())
        q.put(dfb)
    # except:
    #     pass

def f2(q, url):
    try:
    #print("Start: %s" % time.ctime())
        vals = requests.get(url, timeout=4, allow_redirects=False).elapsed.total_seconds()
        g = Goose()
        article = g.extract(url=url)
        text = article.cleaned_text
        blob = TextBlob(text)
        taal = blob.detect_language()
        if taal == ('en'):
            try:
                s = Textatistic(text)
                cols = {
                                'wordcount': [s.word_count],
                                'reponsetime': [vals],
                                'subjectivity': [blob.sentiment.subjectivity],
                                'polarity': [blob.sentiment.polarity],
                                'fleschscore': [s.flesch_score],
                                # 'kw': [ kw ] ,
                                'url': [str(url)]}
                dfa = pd.DataFrame.from_dict(cols)
                        #print(dfa)
                        #print("Start: %s" % time.ctime())
                q.put(dfa)
            except:
                cols = {
                    'wordcount': [str('err')],
                    'reponsetime': [str('err')],
                    'subjectivity': [str('err')],
                    'polarity': [str('err')],
                    'fleschscore': [str('err')],
                    # 'kw': [ kw ] ,
                    'url': [str(url)]}
                dfa = pd.DataFrame.from_dict(cols)
                # print(dfa)
                # print("Start: %s" % time.ctime())
                q.put(dfa)
    except:
        #s = Textatistic(text)
        cols = {
                        'wordcount': [str('err')],
                        'reponsetime': [str('err')],
                        'subjectivity': [str('err')],
                        'polarity': [str('err')],
                        'fleschscore': [str('err')],
                        # 'kw': [ kw ] ,
                        'url': [str(url)]}
        dfa = pd.DataFrame.from_dict(cols)
                #print(dfa)
                #print("Start: %s" % time.ctime())
        q.put(dfa)
        #pass
            #     pass
    #pass


def f3(q, url):
    # try:
    #print("Start: %s" % time.ctime())
    # Instead of returning the result we put it in shared queue.
        st = "/&callback=process&key=57bf606e01a24537ac906a86dc27891f94a0f587"
        # zz = urlopen ( url )
        quez = 'http://api.mywot.com/0.4/public_link_json2?hosts=' + url + st
        stt = urllib.request.urlopen(quez).read()
        stt = str(stt)
        wot = re.findall('\d+', stt)
        ##z=[[conv(s) for s in line.split()] for line in wot]
        z = [conv(s) for s in wot]
        high = (z[1])
        low = (z[2])
        # print ( high , low )
        # WAYBACK
        # zz = "{0.scheme}://{0.netloc}/".format(urlsplit(url))
        # zurlz = "https://web.archive.org/web/0/" + str(zz)
        # r = requests.get(zurlz, allow_redirects=False)
        # data = r.content
        # years = re.findall('\d+', str(data))
        # years = [conv(s) for s in years]
        # years = (years[0])
        # years = int(str(years)[:4])
        cols = {#'yeararchive': [years],
                'lowwot': [low],
                'highwot': [high],
                #'reponsetime': [vals],
                'url': [str(url)]}
        dfb = pd.DataFrame.from_dict(cols)
        #print(dfb)
        #print("Start: %s" % time.ctime())
        q.put(dfb)

def tmpFunc(df):
    #delayed_results = []
    for row in df.itertuples():
        #try:
            url=row.url
            result_queue = Queue()

            # One Thread for response time
            t1 = Thread(target=f1, args=(result_queue, url)) # internet age
            t2 = Thread(target=f2, args=(result_queue, url)) #textual stats
            t3 = Thread(target=f3, args=(result_queue, url)) #mywot

            # Starting threads...
            #print("Start: %s" % time.ctime())
            t1.start()
            t2.start()
            t3.start()

            # Waiting for threads to finish execution...
            t1.join()
            t2.join()
            t3.join()
            #t.join()
            #print("End:   %s" % time.ctime())

            # After threads are done, we can read results from the queue.
            if not result_queue.empty():
                try:
                    r2 = result_queue.get(f2)
                    r1 = result_queue.get(f1)
                    r3 = result_queue.get(f3)
                    #r4 = result_queue.get(f4)
                    #print('slot 1')
                        #print(r1)
                        #print(r1)
                    #print('Slot2')
                        #print(r2)
                    #mergen
                    #try:
                    #df=pd.merge(r1, r2, on='url')
                    dfs = [r1, r2, r3]
                    df = functools.reduce(lambda left, right: pd.merge(left, right, on='url'), dfs)
                    #df = df[['mean', 4, 3, 2, 1]]
                    #print('dss')
                    #print(dss)
                    #print('struct')

                    df = df[['fleschscore', 'highwot', 'lowwot', 'polarity', 'reponsetime', 'subjectivity', 'url', 'wordcount', 'yeararchive']]
                    #print (df)
                #return df
                except:

                    pass
                #print('df')
    return df



# def applyParallel(dfGrouped, func):
#     retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in dfGrouped)
#     return pd.concat(retLst)
def applyParallel(dfGrouped, func):
    with Pool(cpu_count()) as p:
        #retLst = p.map(func, [group for name, group in dfGrouped])
        retLst = p.map(func, [group for name, group in dfGrouped])
    return pd.concat(retLst)



# def applyParallel(dfGrouped, func):
#     retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in dfGrouped)
#     return pd.concat(retLst)

def create_img_url(par_precision, par_accuracy, par_completeness, par_neutrality, par_relevance, par_readability, par_trustworthiness):
    return('<img src=\"../fig/'+str("{0:.0f}".format(par_precision))+'/'+str("{0:.0f}".format(par_accuracy))+'/'+str("{0:.0f}".format(par_completeness))+'/'+str("{0:.0f}".format(par_neutrality))+'/'+str("{0:.0f}".format(par_relevance))+'/'+str("{0:.0f}".format(par_readability))+'/'+str("{0:.0f}".format(par_trustworthiness))+'\"/>')


@app.route('/getterm', methods=['POST', 'GET'])
def get_term():
    if request.method == 'POST':
        tag = request.form['srch-term']
        tt = tag

        mg = MagicGoogle()
        lijst = []
        #tt = 'Donald Trump'
        search = str(tt+' language:english file:html')
        for url in mg.search_url(query=search):
            lijst.append(url)

        df = pd.DataFrame({'url': lijst})
        dff = ((applyParallel(df.groupby(df.index), tmpFunc)))
        #dff = dff[dff.wordcount != 'err']
        newX = dff.values
        newX = np.delete(newX, [6], axis=1)
        pickle_fname = 'pickle.model'
        pickle_model = pickle.load(open(pickle_fname, 'rb'))
        try:
            result = pickle_model.predict(newX)
        except:
            result = np.array([0,0,0,0,0,0,0,0])
        print (result)
        px2 = result.reshape((-1, 8))
        dffres = pd.DataFrame(
            {'OverallQuality': px2[:, 0], 'accuracy': px2[:, 1], 'completeness': px2[:, 2], 'neutrality': px2[:, 3],
             'relevance': px2[:, 4], 'trustworthiness': px2[:, 5], 'readability': px2[:, 6], 'precision': px2[:, 7]})
             #print(dffres)
        print(dffres)
        for row in dffres:
            print(dffres[row])
        dffres2 = {row:plotpie(dffres[row][7],dffres[row][1],dffres[row][2],dffres[row][3],dffres[row][4],dffres[row][6],dffres[row][5])  for row in dffres} #
#dffres2 = {'a':plotpie(1,2,3,4,5,6,7)}
        pd.set_option('display.max_colwidth', -1)
        #print(dffres3)
        #print(type(dffres2))
        #print(dffres3.dtypes)
        return render_template('mp.html', dataframe=dff.to_html(index=False), res=dffres2)


@app.route('/')
def home():
	return render_template('index.html')	
	
@app.route('/pres')
def pres():
    return redirect(url_for('static', filename='pres.html'))
    
@app.route('/getscore',methods=['POST','GET'])
def get_score():
    if request.method=='POST':
        tag = request.form['query']
        url=tag
        g = Goose ( )
        article = g.extract ( url=url )
        text = article.cleaned_text
        blob = TextBlob ( text )
        s = Textatistic ( text )
        vals = requests.get ( url , timeout=4 , allow_redirects=False ).elapsed.total_seconds ( )
        st = "/&callback=process&key=57bf606e01a24537ac906a86dc27891f94a0f587"
        # zz = urlopen ( url )
        quez = 'http://api.mywot.com/0.4/public_link_json2?hosts=' + url + st
        stt = urllib.request.urlopen ( quez ).read ( )
        stt = str ( stt )
        wot = re.findall ( '\d+' , stt )
        ##z=[[conv(s) for s in line.split()] for line in wot]
        z = [ conv ( s ) for s in wot ]
        high = (z[ 1 ])
        low = (z[ 2 ])
        #print ( high , low )
        # WAYBACK
        zz = "{0.scheme}://{0.netloc}/".format ( urlsplit ( url ) )
        zurlz = "https://web.archive.org/web/0/" + str ( zz )
        r = requests.get ( zurlz , allow_redirects=False )
        data = r.content
        years = re.findall ( '\d+' , str ( data ) )
        years = [ conv ( s ) for s in years ]
        years = (years[ 0 ])
        years = int ( str ( years )[ :4 ] )
        cols = {'yeararchive': [ years ] ,
                'lowwot': [ low ] ,
                'highwot': [ high ] ,
                'reponsetime': [ vals ] ,
                'wordcount': [ s.word_count ] ,
                'subjectivity': [ blob.sentiment.subjectivity ],
                'polarity': [ blob.sentiment.polarity ] ,
                'fleschscore': [ s.flesch_score ],
                #'kw': [ kw ] ,
                'url': [ url ]}
        dfeat = pd.DataFrame.from_dict ( cols )
        #df.to_csv ( 'ft.csv' , index=False , sep=',' , encoding='utf-8' )
        del dfeat[ 'url' ]
        #print (df)
        newX = dfeat.values
        pickle_fname = 'pickle.model'
        pickle_model = pickle.load ( open ( pickle_fname , 'rb' ) )
        result = pickle_model.predict ( newX )        #print (result)
        px2 = result.reshape((-1,8))
        dfres = pd.DataFrame({'OverallQuality':px2[:,0],'accuracy':px2[:,1],'completeness':px2[:,2],'neutrality':px2[:,3],'relevance':px2[:,4],'trustworthiness':px2[:,5],'readability':px2[:,6],'precision':px2[:,7]})
        tp = str(keywords(text, words=2))
        # comm = re.compile ( r"https?://(www\.)?" )
        # new_url = comm.sub ( '' , url ).strip ( ).strip ( '/' )
        # print (new_url)
        twtext = list ( )
        polar = list ( )
        datum = list ( )
        for tweet in query_tweets ( tp , 10 ):
            try:
                txt = tweet.text
                txt = re.sub ( r"http\S+" , "" , txt )
                dat = tweet.timestamp
                tblob = TextBlob ( txt )
                tpol = tblob.sentiment.polarity
                tal = tblob.detect_language()
                if tal == ('en'):
                    twtext.append ( txt )
                    polar.append ( tpol )
                    datum.append ( dat )
                else:
                    pass
            except:
                pass
    
    
        df = pd.DataFrame ( {'tweet': twtext , 'timestamp': datum , 'polarity': polar} )
        df[ 'timestamp' ] = pd.to_datetime ( df[ 'timestamp' ] )
        oldest = df[ 'timestamp' ].min ( )
        newest = df[ 'timestamp' ].max ( )
        total = (oldest - newest).total_seconds ( )
        gem = total / len ( df.index )
        #df.to_csv ( 'sentiment.csv' , index=False , sep=',' , encoding='utf-8' )
        tmean = df[ "polarity" ].mean ( )
        tsd = df[ "polarity" ].std ( )
        tkur = df[ "polarity" ].kurtosis ( )
        #topics
        # compile sample documents into a list
        tokenizer = RegexpTokenizer ( r'\w+' )
        stop = set ( stopwords.words ( 'english' ) )
        p_stemmer = PorterStemmer ( )
        doc_set = twtext
        texts = [ ]
    
        for i in doc_set:
            raw = i.lower ( )
            tokens = tokenizer.tokenize ( raw )
            stopped_tokens = [ i for i in tokens if not i in stop ]
            stemmed_tokens = [ p_stemmer.stem ( i ) for i in stopped_tokens ]
            texts.append ( stemmed_tokens )
        dictionary = corpora.Dictionary ( texts )
        corpus = [ dictionary.doc2bow ( text ) for text in texts ]
    
        ldamodel = gensim.models.ldamodel.LdaModel ( corpus , num_topics=1 , id2word=dictionary, minimum_phi_value=0.05)
        topic = ldamodel.print_topics ( num_topics=1, num_words = 1 )
        ctweets = {'meansentiment': [ tmean ] ,
                'sdpolarity': [ tsd ] ,
                'kurtosispolarity': [ tkur ] ,
                'tweetrate': [ gem ] ,
                'tweetcount': [ len ( df.index ) ] ,
                'topic': [ topic ] ,
                'url': [ url ]}
        dftwit = pd.DataFrame.from_dict ( ctweets )
        #entit
        my_sent = article.cleaned_text
        parse_tree = nltk.ne_chunk ( nltk.tag.pos_tag ( my_sent.split ( ) ) , binary=True )  # POS tagging before chunking!
        named_entities = [ ]
        for t in parse_tree.subtrees ( ):
            if t.label ( ) == 'NE':
                named_entities.append ( t )
        z = named_entities
        my_count = pd.Series ( z ).value_counts ( )
        df = pd.DataFrame ( my_count )
        df.columns = [ 'Count' ]
        df[ 'entity' ] = df.index
        za = df.assign ( entity=[ ', '.join ( [ x[ 0 ] for x in r ] ) for r in df.entity ] )
        df[ 'entities' ] = pd.DataFrame ( za[ 'entity' ] )
        del df[ 'entity' ]
        var_input = article.cleaned_text
        var_input = re.sub ( r'[\W\s\d]' , ' ' , var_input )
        input_tokenized = word_tokenize ( var_input , "english" )
        filtered_words = [ word for word in input_tokenized if word not in stopwords.words ( 'english' ) ]
    
        emotion_count = [ ]
    
        for i in range ( 0 , len ( filtered_words ) ):
            with open ( 'em.txt' ) as f:
                for line in f:
                    finaline = line.strip ( )
                    keym = re.search ( "'" + filtered_words[ i ] + "':\s'" , finaline )
                    if keym:
                        # print(keym)
                        valuem = re.findall ( ":\s'.*" , finaline )
                        newstr = str ( valuem )
                        finalvalue = re.sub ( r'[\W\s]' , ' ' , newstr )
                        emotion_count.append ( finalvalue.strip ( ) )
    
        emo = most_common ( emotion_count )
        # tp = str ( keywords ( var_input , words=2 ) )
        tijd = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        col2 = {'emotions': [ emo ] ,
                'topics': [ tp ] ,
                 'tittle': [ article.title ] ,
                 'published': [ article.publish_date ] ,
                 'authors': [ article.authors ] ,
                 'timestamp(gmtime)': [ tijd ] ,
                'url': [ url ]}
        df2 = pd.DataFrame.from_dict ( col2 )
    return render_template('tabs.html', dataframe=dfeat.to_html(index=False) , res=dfres.to_html(index=False), twit=dftwit.to_html(index=False), ent=df.to_html(index=False), des=df2.to_html(index=False))

    
#@app.route('/article')
#def article():
	#return tory(app.root_path + '/../static/', filename)    



#
@app.route('/article') # this is a job for GET, not POST
def article():
	return send_file('static/notfinalized.pdf',
                     mimetype='application/pdf',
                     attachment_filename='notfinalized.pdf',
                     as_attachment=True)	


@app.route('/feedback')
def index():
    OverallQuality_list = ['1', '2', '3', '4', '5']
    accuracy_list = ['1', '2', '3', '4', '5']
    completeness_list = ['1', '2', '3', '4', '5']
    neutrality_list = ['1', '2', '3', '4', '5']
    precision_list = ['1', '2', '3', '4', '5']
    readibility_list = ['1', '2', '3', '4', '5']
    relevance_list = ['1', '2', '3', '4', '5']
    trustworthiness_list = ['1', '2', '3', '4', '5']
    
    return render_template('feedback.html', OverallQuality_list=OverallQuality_list, accuracy_list=accuracy_list, completeness_list=completeness_list, neutrality_list=neutrality_list, precision_list=precision_list, readibility_list=readibility_list, relevance_list=relevance_list, trustworthiness_list=trustworthiness_list)

@app.route('/create', methods=['POST'])
def create_user():
    if request.form['name'] == '':
        flash('Name cannot be blank', 'nameError')
        return redirect('/feedback')
    if request.form['comment'] == '':
        flash('Comment cannot be blank', 'commentError')
        return redirect('//feedback')
    session['comment'] = request.form['comment']
    comment = countLetters(session['comment'])
    print (comment)
    if comment > 120:
        flash('Not more than 120 characters please', 'commentError')
        return redirect('//feedback')

    session['name'] = request.form['name']
    session['OverallQuality'] = request.form['OverallQuality']
    session['accuracy'] = request.form['accuracy']
    session['completeness'] = request.form['completeness']
    session['neutrality'] = request.form['neutrality']
    session['precision'] = request.form['precision']
    session['readibility'] = request.form['readibility']
    session['relevance'] = request.form['relevance']
    session['trustworthiness'] = request.form['trustworthiness']
    tijd = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
       # tijd = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    col2 = {'OverallQuality': (session['OverallQuality']),
             'accuracy': (session['accuracy']) ,
             'completeness': (session['completeness']) ,
             'neutrality':(session['neutrality']) ,
             'precision': (session['precision']) ,
             'readibility': (session['readibility']) ,
             'relevance': (session['relevance']) ,
             'trustworthiness': (session['trustworthiness']),
             'comment': (session['comment']),             
                'timestamp(gmtime)': [ tijd ]}
    fe = pd.DataFrame.from_dict ( col2 )
    if not os.path.isfile('feed.csv'):
        fe.to_csv('feed.csv', index=False)
    else: # else it exists so append without writing the header
        fe.to_csv('feed.csv',mode = 'a',header=False, index=False)
    return redirect('/process')

from io import StringIO, BytesIO

@app.route('/fig/<par_precision>/<par_accuracy>/<par_completeness>/<par_neutrality>/<par_relevance>/<par_readability>/<par_trustworthiness>/')
def plotpie(par_precision, par_accuracy, par_completeness, par_neutrality, par_relevance, par_readability, par_trustworthiness):
    precision = float(par_precision)
    accuracy = float(par_accuracy)
    completeness = float(par_completeness)
    neutrality = float(par_neutrality)
    relevance = float(par_relevance)
    readability = float(par_readability)
    trustworthiness = float(par_trustworthiness)
    group_names=['Precision','Accuracy','Completeness','Neutrality','Relevance','Readability','Trustworthiness']
    group_size=[14.28,14.28,14.28,14.28,14.28,14.28,14.28]
    
    one=['1','1','1','1','1','1','1']
    two=['2','2','2','2','2','2','2']
    three=['3','3','3','3','3','3','3']
    four=['4','4','4','4','4','4','4']
    five=['5','5','5','5','5','5','5']
    
    subgroup_size=[4,4,4,4,4,4,4]
    
    a,b,c,d,e,f,g=[plt.cm.Greys, plt.cm.Purples, plt.cm.Blues, plt.cm.Greens, plt.cm.Oranges, plt.cm.Reds, plt.cm.PuBuGn]
    
    fig, ax=plt.subplots()
    ax.axis('equal')
    mypie, _ = ax.pie(group_size, radius=1.25, labels=group_names, colors=[a(0.6>=(precision>=5)),b(0.6*(accuracy>=5)),c(0.6*(completeness>=5)),d(0.6*(neutrality>=5)),e(0.6*(relevance>=5)),f(0.6*(readability>=5)),g(0.4*(trustworthiness>=5))])
    plt.setp(mypie, width=0.3, edgecolor='black')
    # plt.legend(mypie, group_names, loc="left")
    
    #second Ring inside
    mypie2,_ = ax.pie(subgroup_size, radius=1.25-0.25, labels=four, labeldistance=0.9, colors=[a(0.6>=(precision>=4)),b(0.6*(accuracy>=4)),c(0.6*(completeness>=4)),d(0.6*(neutrality>=4)),e(0.6*(relevance>=4)),f(0.6*(readability>=4)),g(0.4*(trustworthiness>=4))])
    plt.setp(mypie2, width=0.3, edgecolor='black')
    plt.margins(0,0)
    
    #third Ring inside
    mypie3,_ = ax.pie(subgroup_size, radius=1.25-0.5,labels=three,  labeldistance=0.9,colors=[a(0.6>=(precision>=3)),b(0.6*(accuracy>=3)),c(0.6*(completeness>=3)),d(0.6*(neutrality>=3)),e(0.6*(relevance>=3)),f(0.6*(readability>=3)),g(0.4*(trustworthiness>=3))])
    plt.setp(mypie3, width=0.3, edgecolor='black')
    plt.margins(0,0)
    
    # #fourth Ring inside
    mypie4,_ = ax.pie(subgroup_size, radius=1.25-0.75, labels=two, labeldistance=0.8,colors=[a(0.6>=(precision>=2)),b(0.6*(accuracy>=2)),c(0.6*(completeness>=2)),d(0.6*(neutrality>=2)),e(0.6*(relevance>=2)),f(0.6*(readability>=2)),g(0.4*(trustworthiness>=2))])
    plt.setp(mypie4, width=0.3, edgecolor='black')
    plt.margins(1,1)
    
    # #fith Ring inside
    mypie5,_ = ax.pie(subgroup_size, radius=1.25-1, labels=one, labeldistance=0.8,
                      colors=[a(0.6),b(0.6),c(0.6),d(0.6),e(0.6),f(0.6),g(0.4*(trustworthiness>=1))])
    plt.setp(mypie5, width=0.25, edgecolor='black')
    plt.margins(0,0)
    return Markup(mpld3.fig_to_html(fig))
    # img = BytesIO()
    #fig.savefig(img)
    #img.seek(0)
#return send_file(img, mimetype='image/png')


#~ #    app.run(debug=True)
#    app.run(threaded=True, host="0.0.0.0", port=80)
#s	app.run(processes=3)

#    app.run(host='0.0.0.0', port=80)   #app.run()


        


#    app.run(threaded=True)
#app.run(host='127.0.0.1')
    #app.run(host='0.0.0.0')
#test


