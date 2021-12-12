from flask import Flask, render_template, send_file
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

links = {"downloadJSON": "/downloadjson",  # done
         "downloadIPYNB": "/downloadipynb",  # done
         "MMSD": "/mmsd",  # done
         "loc_att": "/la",  # done
         "subs_country": "/subsCountry",  # done
         "wins": "/wins",  # done
         "goals": "/goals",  # done
         "subs_compare": "/subsCompare",  # done
         "pairplot": "/pairplot"}  # done

app = Flask(__name__)


def render_index(image=None, html_string=None, filters=None, errors=None, current_filter_value=""):
    return render_template("index.html", links=links, image=image, code=time.time(), html_string=html_string,
                           filters=filters, errors=errors, current_filter_value=current_filter_value)


@app.route('/', methods=['GET'])
def main_page():
    text = ["Welcome to my <b>Pandas</b>, <b>PyPlot</b> and <b>Flask</b> Project!", "Enjoy (or not):("]
    return render_index(html_string=text)


@app.route(links["downloadIPYNB"], methods=['GET'])
def download_ipynb():
    return send_file("fifa_data.ipynb", as_attachment=True)


@app.route(links["downloadJSON"], methods=['GET'])
def download_data():
    return send_file("fifa_data.json", as_attachment=True)


@app.route(links["pairplot"], methods=['GET', 'POST'])
def pairplot():
    sns_plot = sns.pairplot(df)
    sns_plot.savefig("static/tmp/pairplot.png")
    return render_index(image=("pairplot.png", "pairplot"))


@app.route(links["MMSD"], methods=['GET'])
def meanMedianStd():
    mean = ['Mean value for <b>temperature</b>: ' + str(df['temp_celsius'].mean())]
    median = ['Median value for <b>temperature</b>: ' + str(df['temp_celsius'].median())]
    std = ['Standard deviation for <b>temperature</b>: ' + str(df['temp_celsius'].std())]
    infoTemp = mean + median + std
    mean = ['Mean value for number of <b>events</b>: ' + str(df['event_n'].mean())]
    median = ['Median value for number of <b>events</b>: ' + str(df['event_n'].median())]
    std = ['Standard deviation for number of <b>events</b>: ' + str(df['event_n'].std())]
    infoEvent = mean + median + std
    mean = ['Mean value for number of <b>goals</b>: ' + str(df['goals_n'].mean())]
    median = ['Median value for number of <b>goals</b>: ' + str(df['goals_n'].median())]
    std = ['Standard deviation for number of <b>goals</b>: ' + str(df['goals_n'].std())]
    infoGoals = mean + median + std
    info = infoTemp + ["<hr>"] + infoEvent + ["<hr>"] + infoGoals
    return render_index(html_string=info)


@app.route(links['loc_att'], methods=['GET'])
def locAtt():
    fig = plt.figure(figsize=(30, 10))
    ax = fig.add_subplot(111)
    ax.bar(df['location'], df['attendance'])
    ax.set_title("Dependence of attendance from location")
    plt.xlabel('Location')
    plt.ylabel('Attendance')
    plt.savefig('static/tmp/location&attendance.png')
    return render_index(image=("location&attendance.png", "Dependence of attendance from location"))


@app.route(links['subs_country'], methods=['GET'])
def subsCountry():
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    ax.pie(cntrySum, labels=cntry, rotatelabels=True, autopct='%.2f')
    plt.title('Countries percentage of substitutions', y=1.1)
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


@app.route('/results', methods=['GET', 'POST'])
def about():
    anTwo = ["<h3><u><b>Analysis for Task №2.</b></u></h3>",
             "As we can see on the first graph, Luzhniki Stadium is the most popular "
             "place for watching after game.  "
             "It may be because its the newest stadium, so its more "
             "comfortable. Also, Luzhniki Stadium located in Moscow, Russia, "
             "and Moscow is "
             "the capital.",
             "Let's move to the second graph, Egypt has more substitutions "
             "over all the other countries. I can't tell why but it may be "
             "because of Egypt is one of the smallest country on FIFA, "
             "so there is less money than others to train players, "
             "so they are more weak.", "The last, the third graph. As we can "
                                       "see, the most common outcome is a "
                                       "draw. Then, Croatia and Belgium have "
                                       "the same level of wins."]
    anThree = ["<h3><u><b>Analysis for Task №3.</b></u></h3>",
               "First graph. Belgium did more goals than others and they did it for home "
               "team. After Belgium goes Croatia. However they did more goals as an away "
               "team. England did the same number of goals for home team as for away "
               "team.", "Second graph. England had more substitutions for away team than "
                        "others. Germany had more substitutions for home team. Poland "
                        "did the least number of Poland of substitutions for away team. "
                        "Saudi Arabia did the least number of substitutions for home "
                        "team."]
    html_string = anTwo + ["<hr>"] + anThree
    return render_index(html_string=html_string)


if __name__ == "__main__":
    app.run(debug=True)
