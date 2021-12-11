from flask import Flask, render_template, send_file, request
import time
import pandas as pd
import json
from collections import Counter
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('fifa_data.json', 'r') as f:
    data = json.loads(f.read())

df = pd.DataFrame(data['matches'])
df['attendance'] = df['attendance'].astype(int)
df['attendance'].sum()


def extract_cells(x):
    return int(x['temp_celsius'])


df['temp_celsius'] = df['weather'].apply(extract_cells)

df['event_n'] = df['home_team_events'].apply(len) + df['away_team_events'].apply(len)

df['home_team_goals'] = df['home_team'].apply(lambda x: x['goals'])
df['away_team_goals'] = df['away_team'].apply(lambda x: x['goals'])

c = Counter()

for i, x in df.iterrows():
    c[x['home_team_country']] += x['home_team_goals']
    c[x['away_team_country']] += x['away_team_goals']

h_goals = df.groupby('home_team_country')['home_team_goals'].sum()
a_goals = df.groupby('away_team_country')['away_team_goals'].sum()
(h_goals + a_goals).sort_values(ascending=False)

df['home_team_statistics'].apply(lambda x: x['tactics']).value_counts()


def count_substitutions(ev_arr):
    s = 0
    for v in ev_arr:
        if v['type_of_event'].startswith('substitution'):
            s += 1
    return s


df['home_team_subst'] = df['home_team_events'].apply(count_substitutions)
df['away_team_subst'] = df['away_team_events'].apply(count_substitutions)

df.head()

h_subst = df.groupby('home_team_country', as_index=False)['home_team_subst'].agg(['sum', 'size'])
a_subst = df.groupby('away_team_country', as_index=False)['away_team_subst'].agg(['sum', 'size'])

home_subst = df.groupby('home_team_country', as_index=False)['home_team_subst'].sum()
away_subst = df.groupby('away_team_country', as_index=False)['away_team_subst'].sum()

mean_subst = h_subst + a_subst

mean_subst['mean'] = mean_subst['sum'] / mean_subst['size']

mean_subst.sort_values(by='mean', ascending=False)

df['goals_n'] = df['home_team_goals'] + df['away_team_goals']

cntryWin = dict()
country = list()
wins = list()
for i in df['winner']:
    if i not in cntryWin:
        cntryWin[i] = 1
    else:
        cntryWin[i] += 1
for i in cntryWin.items():
    country.append(i[0])
    wins.append(i[1])
print(len(country))
print(len(wins))
s = len(mean_subst)
cntry = list()
for i in range(s):
    cntry.append(mean_subst.index[i])
k = 0
cntrySum = list()
for i in cntry:
    cntrySum.append(mean_subst[i:]['sum'][0])

htg = df.groupby('home_team_country', as_index=False)['home_team_goals'].sum()
atg = df.groupby('away_team_country', as_index=False)['away_team_goals'].sum()

links = {"download" : "/download", #done
         "MMSD" : "/mmsd", #done
         "loc_att": "/la", #done
         "subs_country": "/subsCountry",#done
         "wins": "/wins", #done
         "goals": "/goals", #done
         "subs_compare": "/subsCompare", #done
         "pairplot": "/pairplot"} #done

app = Flask(__name__)

def render_index (image=None, html_string=None, filters=None,  errors=None, current_filter_value=""):
    return render_template("index.html", links=links, image=image, code=time.time(), html_string=html_string,
                           filters=filters, errors=errors, current_filter_value=current_filter_value)

@app.route('/', methods=['GET'])
def main_page():
    return render_index()

@app.route(links["download"], methods=['GET'])
def download_data():
    return send_file("fifa_data.json", as_attachment=True)

@app.route(links["pairplot"], methods=['GET', 'POST'])
def pairplot():
    sns_plot = sns.pairplot(df)
    sns_plot.savefig("static/tmp/pairplot.png")
    return render_index(image=("pairplot.png", "pairplot"))

@app.route(links["MMSD"], methods=['GET'])
def meanMedianStd():
    mean = ['Mean value for temperature: ' + str(df['temp_celsius'].mean())]
    median = ['Median value for temperature: ' + str(df['temp_celsius'].median())]
    std = ['Standard deviation for temperature: ' + str(df['temp_celsius'].std())]
    infoTemp = mean + median + std
    mean = ['Mean value for number of events: ' + str(df['event_n'].mean())]
    median = ['Median value for number of events: ' + str(df['event_n'].median())]
    std = ['Standard deviation for number of events: ' + str(df['event_n'].std())]
    infoEvent = mean + median + std
    mean = ['Mean value for number of goals: ' + str(df['goals_n'].mean())]
    median = ['Median value for number of goals: ' + str(df['goals_n'].median())]
    std = ['Standard deviation for number of goals: ' + str(df['goals_n'].std())]
    infoGoals = mean + median + std
    info = infoTemp + infoEvent + infoGoals
    return render_index(html_string=info)

@app.route(links['loc_att'], methods=['GET'])
def locAtt():
    fig = plt.figure(figsize=(30, 10))
    ax = fig.add_subplot(111)
    ax.bar(df['location'], df['attendance'])
    plt.xlabel('Location')
    plt.ylabel('Attendance')
    plt.savefig('static/tmp/location&attendance.png')
    return render_index(image=("location&attendance.png", "Dependence of attendance from location"))

@app.route(links['subs_country'], methods=['GET'])
def subsCountry():
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111)
    ax.pie(cntrySum, labels=cntry, rotatelabels=True, autopct='%.2f')
    plt.title('Countries percentage of substitutions', y=-1.3)
    plt.savefig('static/tmp/subs&country.png')
    return render_index(image=("subs&country.png", "Countries and percentage of substitutions"))

@app.route(links['wins'], methods=['GET'])
def win():
    fig = plt.figure(figsize=(32, 10))
    ax = fig.add_subplot(111)
    ax.plot(country, wins, linewidth=2.5)
    ax.set_title('Country wins')
    plt.xlabel('Country')
    plt.ylabel('Wins')
    plt.savefig('static/tmp/wins.png')
    return render_index(image=("wins.png", "Countries' wins"))

@app.route(links['goals'], methods=['GET'])
def goals():
    fig = plt.figure(figsize=(35, 10))
    ax = fig.add_subplot(111)

    ax.bar(htg['home_team_country'], htg['home_team_goals'])
    ax.bar(atg['away_team_country'], atg['away_team_goals'], bottom=htg['home_team_goals'])

    ax.legend(['as a home team', 'as an away team'])
    ax.set_title('Goals for home or away team')
    plt.ylabel('Number of goals')
    plt.xlabel('Country')
    plt.savefig('static/tmp/goals.png')
    return render_index(image=("goals.png", "Country goals"))

@app.route(links['subs_compare'], methods=['GET'])
def subsCompare():
    fig = plt.figure(figsize=(35, 10))
    ax = fig.add_subplot(111)

    ax.plot(home_subst['home_team_country'], home_subst['home_team_subst'], linewidth=2.3)
    ax.plot(away_subst['away_team_country'], away_subst['away_team_subst'], linewidth=2.3)
    ax.scatter(home_subst['home_team_country'], home_subst['home_team_subst'], s=100)
    ax.scatter(away_subst['away_team_country'], away_subst['away_team_subst'], s=100)

    plt.legend(['home team substitution', 'away team substitution'])
    ax.set_title('Number of substitution depending on playing side')
    plt.xlabel('Country')
    plt.ylabel('Number of substitution')
    plt.savefig('static/tmp/subsCompare.png')
    return render_index(image=("subsCompare.png", "Compare of substitutions"))

@app.route('/raw', methods=['GET', 'POST'])
def about():
    with open('fifa_data.json', 'r') as f:
        data = json.loads(f.read())
    dff = pd.DataFrame(data['matches'])
    errors = []
    current_filter_value = ""
    if request.method == "POST":
        current_filter = request.form.get('filters')
        current_filter_value = current_filter
        if current_filter:
            try:
                dff = dff.query(current_filter)
            except Exception as e:
                errors.append('<font color="red">Incorrect filter</font>')
                print(e)

    html_string = dff.to_html()
    return render_index(html_string=html_string, filters=True, errors=errors, current_filter_value=current_filter_value)


if __name__ == "__main__":
    app.run(debug=True)