#source ./eb-dash2/bin/activate

import dash
from dash.dependencies import Input, Output, Event, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime as dt
import datetime as dt
import os
import json
import dateutil
import tweepy
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.figure_factory as ff
#from botHunter import classification
import networkx as nx

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
import classification
import twitter_col
import time


app = dash.Dash(__name__)
application = app.server

def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

with open('.stuff.json', 'r') as f:
    for line in f:
        stuff = json.loads(line)

auth = tweepy.OAuthHandler(stuff['consumer_key'], stuff['consumer_secret'])
auth.set_access_token(stuff['access_token'], stuff['access_secret'])

api = tweepy.API(auth, wait_on_rate_limit=True,
				   wait_on_rate_limit_notify=True)

features = pd.read_csv('features.csv')

markdown_text2 = '''
##  *botHunter* Account Classification

This application allows the user to determine whether a given Twitter account has bot-like characteristics and behaviors.  Enter an active Twitter Screen Name below and press *Submit*.


'''

retweet_markdown = '''

-----

#### Retweet analysis

Bots are often used to propogate a message or amplify another user(s).  This is often done with retweets, since these are easy to automate.  Bots will often have far more retweets than replies or original content.  If a bot does produce original content, it is often non-sensical.



'''

network_markdown = '''
### Network analysis

This network represents the comention of entities (Hashtags, mentions, and URLs) in the given users timeline.  This network helps to identify various groups or conversations that the user participates in.

'''

time_markdown = '''
-----

### Temporal analysis

Bot herders have to set up times when their accounts will execute activity.  This can be event driven, but is often time driven.  The graph below provides tweets per hour in the timeline (up to last 200 tweets).  This helps the user identify any automated timelines.  It also helps to make sure the tweet volume is feasible (i.e. 200 tweets an hour for an extended time period is not likely for a human.)  Try classifying the account @SaltyCorpse to see how you can find a bot by analyzing temporal patterns.



'''

languages_markdown = '''

-----

### Language analysis

The plots below provide insight into the different languages identified in the user timeline.  Some bots try to mask their purpose by randomly retweeting.  This produces a timeline that has many topics and many languages.  Analyzing the distribution of languages can assist in finding bots.



'''

sources_markdown = '''

-----

### Source analysis

Twitter JSON includes the type of platform or the application that was used to generate the tweet.  Below is analysis of the platforms used by the given user.  Some applications are frequented by bots.



'''

line_markdown = '''

*Disclaimer: These models simply provide evidence of whether a given account has characteristics and behavior similar to bots.  Genuine users can at times have accounts with similar characteristics.*

-----

'''

description_markdown = '''


#### Bot-hunter Analysis

Automated social media *bots* have existed almost as long as the social media platforms they inhabit.  Their emergence has triggered numerous research efforts to develop increasingly sophisticated means to detect these accounts.  These efforts have resulted in a *cat and mouse* cycle in which detection algorithms evolve trying to keep up with ever evolving *bots*.

This web dashboard was designed to help researchers explore bot activity as well as the conversations and networks that they inhabit.  This componenet of the dashboard allows the user to enter a Twitter Screen Name, and the app will scrape necessary data from the Twitter API, render a supervised machine learning classification of the account, and then provide exploratory visualization of the data to assist the researcher in recognizing and communicating any automated acitivty that may be present.

#### Question

The overarching question that this dashboard seeks to answer is:

*Does a given Twitter account exhibit bot-like characteristics and behavior?*

This question is answered both with supervised machine learning as well as user driven data exploration.  Sometimes a machine learning model can seem like a black box for a user.  Therefore, in addition to providing the ML prediction, this data narrative allows the user to explore the network features, temporal nature, as well as several other features of the user provided account.


#### Explanation of Data Sources and Data Access

All data is accessed through the Twitter REST API using the [Tweepy](http://www.tweepy.org/) Python Package.  When the user submits the request, the application will collect the *user* JSON object as well as the *user* timeline (last 200 tweets).  The Four C's of Data Quality are Discussed below:

**1. Completeness:** The Data is collected live, and is complete.  Note that the only debatable concern we have is whether 200 tweets from the timeline are sufficient.  We could collect up to the last 3000 tweets, but this would be time and disk intensive.  We believe 200 is sufficient for prediction and exploration.

**2. Coherent:**: The Twitter API is predictable and fairly clean.  Data is as expected.  Note that the free text tweet, however, can be very messy and does not use standard spelling/grammar.  The two errors we had to address with our code is a) what to do if account doesn't exist or is suspended and b) Fix algorithms to work if the user has never posted a single tweet.  These were fixed in the application

**3. Correct:** Note that our prediction will be biased toward the types of bots that we trained our supervised ML algorithm on.  There are many types of bots, and our training set, while diverse and fairly robust, does not encompass all types.

**4. Accountable:** Our data is collected at the moment the user presses *submit* and therefore is timely and accountable.



#### Data Pipeline

1. Collect user JSON and user Timeline JSON from Twitter REST API
2. Determine whether the user has a timeline (has posted at least 1 tweet)
3. Conduct appropriate feature extraction:
    - If timeline present, extract content, profile, temporal, and network features from user object and user timelin
    - If no timeline present, extract content, profile, temporal, and network features only from user object
4. Predict whether account is a bot using Random Forest Classifier in [sklearn](http://scikit-learn.org/stable/)
5. Provide user feedback and render exploratory visualizations

#### Modeling specifics

In my search for the best machine learning algorithm, I test Naive Bayes, Logistic Regression, Support Vector Machines, Decision Trees, and Random Forest Classifier.

I measured the performance of each of these models in 10 fold cross validation.  The performance of each is given below:

1. Naive Bayes: AUC = 0.885
2. SVM: AUC = 0.921
3. Logisitic Regression: AUC = 0.950
4. Decision Tree: AUC = 0.963
5. Random Forest: AUC = 0.994

Our model therefore uses Random Forest Classifier.

Our annotated bot data is derived from 1) 19K [bots that attacked NATO](https://medium.com/dfrlab/botspot-the-intimidators-135244bfe46b) in summer of 2017, 2) Bots annotated through random string detection and 3) Bots from the [Texas A and M honeypot data set](https://pdfs.semanticscholar.org/b433/9952a73914dc7eacf3b8e4c78ce9a5aa9502.pdf). In order to train a model, we also needed accounts that we could tag as *human*, and not automated.  We used the Twitter Streaming API to collect a sample of *normal* Twitter data.  From this data we randomly selected 70,000 accounts to tag as *human* Twitter accounts that we can sample from for training.  [Past research](https://arxiv.org/pdf/1703.03107.pdf?ftag=MSFd61514f) has estimated that 5-8% of twitter accounts are automated.  If this is true, then we mis-labeled a small amount of our accounts as *human*.  We assessed that this was an acceptable amount of noise in the data, but that it will undoubtedly limit the performance of supervised learning models that train and test on the data.

#### Web app design decisions

In building this web applications, we chose to use Plotly Dash, which sits on top of the Flask App framework.  This allowed us to use plotly visualizations.  It also allowed the app to render quickly, since Plotly Dash callbacks execute in parallel.

At the recommendation of the Plotly Dash Team, we built our app with a single page.  While some would say that this is cluttered, this helps the app run much faster than having multiple page.

We chose a few plots that we felt would help a user explore the bot-like characteristics of various accounts.  The reason for each plot is discussed to the left



#### What we learned

This app allowed us to quickly assess the strengths and weaknesses of our prediction algorithm, as well as see the immediate value of allowing a user to explore the data to find evidence of bot activity.

One of the first lessons of the limitations of our bot detection algorithm occured when I found that it predicted my account as well as some of my friends account as bots.  This likely happened because the random forest predictor primarily leverages the friend/follower distributions in the prediction.  If a human account just happens to have friends/followers in the same distribution as bots, then they will become a false positive.

We also quickly saw the value of the exploratory analysis to validate and explore the prediction.  For example, if you put @Bratterina or @SaltyCorpse into the prediction, they both come back as NOT bot-like.  However, if you look at the temporal plot, you can immediately see that these are absolutely bots (they each post 40-80 tweets every night at midnight).  This shows the power of providing concrete and irrefutable evidence through the hands on data exploration.



-----

'''

features = pd.read_csv('features.csv')

app.layout = html.Div([
    # dcc.Markdown(children=markdown_text2),
    #  html.Div(dcc.Input(id='input-box', type='text', value = '@PxXVzkM9oyMfIZY',placeholder='Enter a screen name (i.e. @PxXVzkM9oyMfIZY')),
    #  html.Button(id='submit', n_clicks = 0, type='submit', children='Submit'),
    #  html.H3(id='my-div',style={'color': '#7F90AC'}),
    #   dcc.Markdown(children=line_markdown),

    html.Div([
        html.Div([
            dcc.Markdown(children=markdown_text2),
            html.Div(dcc.Input(id='input-box', type='text', value = '@PxXVzkM9oyMfIZY',placeholder='Enter a screen name (i.e. @PxXVzkM9oyMfIZY')),
            html.Button(id='submit', n_clicks = 0, type='submit', children='Submit'),
            html.H3(id='my-div',style={'color': '#7F90AC'}),
            dcc.Markdown(children=line_markdown),

            dcc.Markdown(children=network_markdown),
            dcc.Graph(id='my-network'),
            dcc.Markdown(children=time_markdown),
            dcc.Graph(id='my-timeseries'),
            dcc.Markdown(children=retweet_markdown),
            dcc.Graph(id='my-retweets'),
            dcc.Markdown(children=languages_markdown),
            dcc.Graph(id='my-languages'),
            dcc.Markdown(children=sources_markdown),
            dcc.Graph(id='my-sources'),
            dcc.Graph(id='my-table')

        ], className="eight columns"),

        html.Div([
            dcc.Markdown(children=description_markdown)
            # generate_table(features)
        ], className="four columns"),
    ], className="row")
])

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})


@app.callback(
             Output(component_id='my-div', component_property='children'),
   [Input('submit', 'n_clicks')], [State('input-box', 'value')]
    )
def update_output_div(n_clicks,value):

    x = n_clicks

    # if value == None:
    #     screen_name = 'PxXVzkM9oyMfIZY'
    # else:
    screen_name = value.lstrip('@')

    try:
        user = api.get_user(screen_name = screen_name)
    except tweepy.error.TweepError as e:
        reason = eval(e.reason)
        return "{}"' Data not available.  Reason: '"{}".format(screen_name,reason[0]['message'])
    with open('user_json/' + screen_name + '.json', 'w') as sn_out:
        out = json.dumps(user._json)
        sn_out.write(out + '\n')

    if user.statuses_count > 10:
        bot = classification.bot_classification_time('user_json/' + screen_name + '.json', 'combined_new_tier2_20180428-232430.pkl', api, 'timelines')
    else:
        bot = classification.bot_classification('user_json/' + screen_name + '.json', 'combined_new_tier1_20180428-220633.pkl')

    my_bot = bot['prediction'][0]

    if my_bot:
        return "{}"'  DOES exhibit bot-like characteristics and behavior'.format(screen_name)
    else:
        return "{}"' DOES NOT exhibit bot-like characteristics and behavior'.format(screen_name)


@app.callback(Output('my-timeseries', 'figure'), [Input('submit', 'n_clicks')], [State('input-box', 'value')])
def update_graph(n_clicks,value):
    screen_name = value.lstrip('@')
    try:
        user = api.get_user(screen_name = screen_name)
        timeline = classification.get_timeline(api, user.id_str, directory = 'timelines')
    except:
        timeline = []
    df = twitter_col.parse_twitter_list(timeline)
    df['date2'] = twitter_col.convert_dates(df['status_created_at'].tolist())
    df.index = pd.DatetimeIndex(df.date2)
    time_series = df['date2'].resample('D').count()
    return {
        'data': [{
            'x': time_series.index,
            'y': time_series
        }],
        'layout': {
                'title': 'Tweets per Hour'
            }
    }

@app.callback(Output('my-retweets', 'figure'),  [Input('submit', 'n_clicks')], [State('input-box', 'value')])
def update_retweets(n_clicks,value):
    screen_name = value.lstrip('@')
    try:
        user = api.get_user(screen_name = screen_name)
        timeline = classification.get_timeline(api, user.id_str, directory = 'timelines')
    except:
        timeline = []
    df = twitter_col.parse_twitter_list(timeline)
    retweets = df['status_isretweet'].sum()
    replies =  df['reply_to_status_id'].count()
    original = len(df.index) - (retweets+replies)
    trace0 = go.Bar(
        x=['Retweets', 'Replies', 'Original Content'],
        y=[retweets,replies, original],
        text=[str(retweets) + ' retweets', str(replies) + ' replies' , str(original) + ' original content tweets'],
        marker=dict(
            color='rgb(162,159,166)',
            line=dict(
                color='rgb(24,27,45)',
                width=1.5,
            )
        ),
        opacity=0.6
    )

    data = [trace0]
    layout = go.Layout(
        title='Comparison of Retweets, Replies, and Original Content',
    )

    fig = go.Figure(data=data, layout=layout)
    return(dict(fig))

@app.callback(Output('my-languages', 'figure'),  [Input('submit', 'n_clicks')], [State('input-box', 'value')])
def update_languages(n_clicks,value):
    screen_name = value.lstrip('@')
    try:
        user = api.get_user(screen_name = screen_name)
        timeline = classification.get_timeline(api, user.id_str, directory = 'timelines')
    except:
        timeline = []
    df = twitter_col.parse_twitter_list(timeline)
    languages = df['status_lang'].value_counts()
    trace0 = go.Bar(
        x=languages.index.tolist(),
        y=languages.tolist(),
        text=[languages.tolist()],
        marker=dict(
            color='rgb(162,159,166)',
            line=dict(
                color='rgb(24,27,45)',
                width=1.5,
            )
        ),
        opacity=0.6
    )

    data = [trace0]
    layout = go.Layout(
        title='Sources Used by ' + screen_name,
    )

    fig = go.Figure(data=data, layout=layout)
    return(dict(fig))

@app.callback(Output('my-sources', 'figure'),  [Input('submit', 'n_clicks')], [State('input-box', 'value')])
def update_sources(n_clicks,value):
    screen_name = value.lstrip('@')
    try:
        user = api.get_user(screen_name = screen_name)
        timeline = classification.get_timeline(api, user.id_str, directory = 'timelines')
    except:
        timeline = []
    df = twitter_col.parse_twitter_list(timeline)
    languages = df['status_source'].value_counts()
    trace0 = go.Bar(
        x=languages.index.tolist(),
        y=languages.tolist(),
        text=[languages.tolist()],
        marker=dict(
            color='rgb(162,159,166)',
            line=dict(
                color='rgb(24,27,45)',
                width=1.5,
            )
        ),
        opacity=0.6
    )

    data = [trace0]
    layout = go.Layout(
        title='Sources Used by ' + screen_name,
    )

    fig = go.Figure(data=data, layout=layout)
    return(dict(fig))

@app.callback(Output('my-network', 'figure'),  [Input('submit', 'n_clicks')], [State('input-box', 'value')])
def update_network(n_clicks,value):
    screen_name = value.lstrip('@')
    try:
        user = api.get_user(screen_name = screen_name)
        timeline = classification.get_timeline(api, user.id_str, directory = 'timelines')
    except:
        timeline = []
    if len(timeline) > 0:
        edge = classification.extract_entity_comention(timeline)
        G=nx.from_pandas_edgelist(edge, 'entity1', 'entity2')

    # G=nx.random_geometric_graph(200,0.125)
    # G=nx.complete_graph(100)
#    G = nx.erdos_renyi_graph(100, 0.125)
        pos = nx.spring_layout(G)
        nx.set_node_attributes(G, pos, 'pos')
        degree = G.degree()
        nx.set_node_attributes(G, dict(degree), 'degree')

        dmin=1
        ncenter=0
        for n in pos:
            x,y=pos[n]
            d=(x-0.5)**2+(y-0.5)**2
            if d<dmin:
                ncenter=n
                dmin=d

        p=nx.single_source_shortest_path_length(G,ncenter)

    edge_trace = Scatter(
        x=[],
        y=[],
        line=Line(width=0.5,color='#888'),
        hoverinfo='none',
        mode='lines')

    if len(timeline) > 0:
        for edge in G.edges():
            x0, y0 = G.node[edge[0]]['pos']
            x1, y1 = G.node[edge[1]]['pos']
            edge_trace['x'] += [x0, x1, None]
            edge_trace['y'] += [y0, y1, None]

    node_trace = Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=Marker(
            showscale=True,
            # colorscale options
            # 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' |
            # Jet' | 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'
            colorscale='YIGnBu',
            reversescale=True,
            color=[],
            size=6,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))
    if len(timeline) > 0:
        mat = nx.adjacency_matrix(G)
        Mc = mat.tocoo()
        mat = [Mc.col.tolist(), Mc.data.tolist()]

        for node in G.nodes():
            x, y = G.node[node]['pos']
            node_trace['x'].append(x)
            node_trace['y'].append(y)

        for node in G.nodes():
            node_trace['marker']['color'].append(G.node[node]['degree'])
            node_info = node + ':<br> # of connections: '+str(G.node[node]['degree'])
            node_trace['text'].append(node_info)

    fig = Figure(data=Data([edge_trace, node_trace]),
             layout=Layout(
                title='<br>Network graph made with Python',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Python code: <a href='https://plot.ly/ipython-notebooks/network-graphs/'> https://plot.ly/ipython-notebooks/network-graphs/</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))
    return(dict(fig))

@app.callback(Output('my-table', 'figure'),  [Input('submit', 'n_clicks')], [State('input-box', 'value')])
def update_table(n_clicks, value):
    screen_name = value.lstrip('@')
    try:
        user = api.get_user(screen_name = screen_name)
        timeline = classification.get_timeline(api, user.id_str, directory = 'timelines')
    except:
        timeline = []

    if len(timeline) > 0:
        df = classification.create_feature_space_with_time('user_json/' + screen_name + '.json',  api, 'timelines')
    else:
        df = classification.create_feature_space('user_json/' + screen_name + '.json')
    df = df.transpose()
    df.columns.values[0] = 'Value'
    df['Feature'] = df.index
    final = pd.merge(features, df, how = 'inner', on = 'Feature')
    # return(generate_table(final, 50))
    return(ff.create_table(final))

if __name__ == '__main__':
    # app.run_server(debug = True)
    application.debug = True
    application.run()
