import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AffinityPropagation
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import plotly as py
import plotly.graph_objs as go
import os
import datetime as dt
import missingno as msno
import streamlit as st

st.set_page_config(layout="wide")

df=pd.read_csv('project_sample/data/netflix_titles.csv')

#replacement
df['country'] = df['country'].fillna(df['country'].mode()[0])
df['cast'].replace(np.nan, 'No Data', inplace=True)
df['director'].replace(np.nan, 'No Data' , inplace=True)

#Drops
df.dropna(inplace=True)

#Drop Duplicate
df.drop_duplicates(inplace=True)

df['date_added'] =pd.to_datetime(df['date_added'].str.strip(), errors='coerce')
df['month_added']=df['date_added'].dt.month
df['month_name_added'] = df['date_added'].dt.month_name()
df['year_added']=df['date_added'].dt.year

with st.container():
    st.write('## Data Frame Netflix Analysis')
    st.dataframe(df.head(10), width= None, use_container_width=True)

#pallet
sns.palplot(['#221f1f', '#b20710', '#e50914','#f5f5f1'])
plt.title("Netflix brand palette ",loc='left',fontfamily='serif',fontsize=15,y=1.2)

# movie and  Tv Show distrbution

#ratio of movie and Tv Show
df_group_type = df.groupby(['type'])['type'].count()
df_count = len(df)
ratio = ((df_group_type/df_count)).round(2)

df_ration = pd.DataFrame(ratio).T


# Visualisasi Analisa
st.html(
    "<h1><span style="
    "'color:black; font-family:Arial; font-size:30px; font-weight:bold; padding:0px;'"
    ">Visualisasi Analysis</span></h1>"
)

col1, col2 = st.columns(2)
with col1 :
    #Visualisasi Ratio
    fig, ax = plt.subplots(1,1, figsize=(7.5 , 1.5))

    ax.barh (df_ration.index, df_ration['Movie'],
         color='#b20710', alpha = 0.9, label='Male')
    ax.barh (df_ration.index, df_ration['TV Show'], left=df_ration['Movie'],
         color='#221f1f', alpha = 0.9, label='Female')

    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_yticklabels(mf_ratio.index, fontfamily='serif', fontsize=11)

    #movie Persent
    for i in df_ration.index:
        ax.annotate(f"{int(df_ration['Movie'][i]*100)}%",
                xy=(df_ration['Movie'][i]/2,i),
                va='center', ha='center', fontsize=40, fontweight='light', fontfamily='serif', color='white')
        ax.annotate("Movie",
                xy=(df_ration['Movie'][i]/2, -0.25),
                va ='center', ha='center', fontsize='15', fontweight='light', fontfamily='serif', color='white')
    #TvShow Persent
    for i in df_ration.index:
        ax.annotate(f"{int(df_ration['TV Show'][i]*100)}%",
                xy=(df_ration['Movie'][i]+df_ration['TV Show'][i]/2,i),
                va='center', ha='center', fontsize=40, fontweight='light', fontfamily='serif', color='white')
        ax.annotate("TV Show",
                xy=(df_ration['Movie'][i]+df_ration['TV Show'][i]/2, -0.25),
                va ='center', ha='center', fontsize='15', fontweight='light', fontfamily='serif', color='white')

    ax.legend().set_visible(False)
    
    st.pyplot(fig)

with col2 :
    st.write("""##  Movie & TV Show Distribution
    Sebagian besar konten Netflix merupakan Movie dibandingkan TV Show 
    Grafik ini menggambarkan proporsi distribusi masing-masing.""")

#VISUALISASI BY CONTENT COUNTRY
# Helper column for various plots
df['count']=1

# Many productions have several countries listed - this will skew our results , we'll grab the first one mentioned
# Lets retrieve just the first country
df['first_country'] = df['country'].apply(lambda x :x.split(",")[0])
# Rating ages from this notebook: https://www.kaggle.com/andreshg/eda-beginner-to-expert-plotly (thank you!)
ratings_ages = {
    'TV-PG': 'Older Kids',
    'TV-MA': 'Adults',
    'TV-Y7-FV': 'Older Kids',
    'TV-Y7': 'Older Kids',
    'TV-14': 'Teens',
    'R': 'Adults',
    'TV-Y': 'Kids',
    'NR': 'Adults',
    'PG-13': 'Teens',
    'TV-G': 'Kids',
    'PG': 'Older Kids',
    'G': 'Kids',
    'UR': 'Adults',
    'NC-17': 'Adults'
}
df['target_ages'] = df['rating'].replace(ratings_ages)
df['target_ages'].unique()

# Genre

df['genre'] = df['listed_in'].apply(lambda x :  x.replace(' ,',',').replace(', ',',').split(',')) 

# Reducing name length

df['first_country'].replace('United States', 'USA', inplace=True)
df['first_country'].replace('United Kingdom', 'UK',inplace=True)
df['first_country'].replace('South Korea', 'S. Korea',inplace=True)

df_group_country = df.groupby('first_country')['count'].sum().sort_values(ascending=False)[:10]

# Plot

color_map = ['#f5f5f1' for _ in range(10)]
color_map[0] = color_map[1] = color_map[2] =  '#b20710' # color highlight

col1, col2 = st.columns(2)
with col1 :
    st.write(""" ### Top 10 countries on Netflix
The three most frequent countries have been highlighted.
The most prolific producers of content for Netflix are, primarily,
the USA, with India and the UK a significant distance behind 
It makes sense that the USA produces the most content as, afterall, 
Netflix is a US company""")
    fig, ax = plt.subplots(1,1, figsize=(12, 6))
    ax.bar(df_group_country.index, df_group_country, width=0.5, 
       edgecolor='darkgray',
       linewidth=0.6,color=color_map)
    
    #menampilkan nilai di atas bar
    for i in df_group_country.index:
        ax.annotate(f"{df_group_country[i]}", 
                   xy=(i, df_group_country[i] + 50), #i like to change this to roughly 5% of the highest cat
                   va = 'center', ha='center',fontweight='light', fontfamily='serif')
    
    #menghilangkan border di pinggir
    for s in ['top', 'left', 'right']:
        ax.spines[s].set_visible(False)
    
    # Tick labels
    ax.set_xticklabels(df_group_country.index, fontfamily='serif', rotation=0)

    

    st.pyplot(fig)

    
with col2 :
        st.write(""" ### Top 10 countries Movie & TV Show split
    Percent Stacked Bar Chart.
    Interestingly, Netflix in India is made up nearly entirely of Movies.
    Bollywood is big business, and perhaps the main focus of this industry is Movies and not TV Shows.
    South Korean Netflix on the other hand is almost entirely TV Shows
                 """)
         
        country_order = df['first_country'].value_counts()[:11].index
        data_q2q3 = df[['type', 'first_country']].groupby('first_country')['type'].value_counts().unstack().loc[country_order]
        data_q2q3['sum'] = data_q2q3.sum(axis=1)
        data_q2q3_ratio = (data_q2q3.T / data_q2q3['sum']).T[['Movie', 'TV Show']].sort_values(by='Movie',ascending=False)[::-1]

        ###
        fig, ax = plt.subplots(1,1,figsize=(15, 8),)

        ax.barh(data_q2q3_ratio.index, data_q2q3_ratio['Movie'], 
        color='#b20710', alpha=0.8, label='Movie')

        ax.barh(data_q2q3_ratio.index, data_q2q3_ratio['TV Show'], left=data_q2q3_ratio['Movie'], 
        color='#221f1f', alpha=0.8, label='TV Show')

        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_yticklabels(data_q2q3_ratio.index, fontfamily='serif', fontsize=11)

        # male percentage
        for i in data_q2q3_ratio.index:
            ax.annotate(f"{data_q2q3_ratio['Movie'][i]*100:.3}%", 
                   xy=(data_q2q3_ratio['Movie'][i]/2, i),
                   va = 'center', ha='center',fontsize=12, fontweight='light', fontfamily='serif',
                   color='white')

        for i in data_q2q3_ratio.index:
            ax.annotate(f"{data_q2q3_ratio['TV Show'][i]*100:.3}%", 
                   xy=(data_q2q3_ratio['Movie'][i]+data_q2q3_ratio['TV Show'][i]/2, i),
                   va = 'center', ha='center',fontsize=12, fontweight='light', fontfamily='serif',
                   color='white')
        st.pyplot(fig)

st.markdown("<br><br>", unsafe_allow_html=True)

# VISUALISASI RATING

order = pd.DataFrame(df.groupby('rating')['count'].sum().sort_values(ascending=False).reset_index())
rating_order = list(order['rating'])

mf = df.groupby('type')['rating'].value_counts().unstack().sort_index().fillna(0).astype(int)[rating_order]
movie = mf.loc['Movie']
tv = - mf.loc['TV Show']

fig, ax = plt.subplots(1,1, figsize=(20, 3))
ax.bar(movie.index, movie, width=0.5, color='#b20710', alpha=0.8, label='Movie')
ax.bar(tv.index, tv, width=0.5, color='#221f1f', alpha=0.8, label='TV Show')
# Annotations
for i in tv.index:
    ax.annotate(f"{-tv[i]}", 
                   xy=(i, tv[i] - 90),
                   va = 'center', ha='center',fontweight='light', fontfamily='serif',
                   color='#4a4a4a')   

for i in movie.index:
    ax.annotate(f"{movie[i]}", 
                   xy=(i, movie[i] + 90),
                   va = 'center', ha='center',fontweight='light', fontfamily='serif',
                   color='#4a4a4a')

for s in ['top', 'left', 'right', 'bottom']:
    ax.spines[s].set_visible(False)

ax.set_xticklabels(mf.columns, fontfamily='serif')
ax.set_yticks([]) 

ax.legend().set_visible(False)
fig.text(0.16, 1.1, 'Rating distribution by Film & TV Show', fontsize=15, fontweight='bold', fontfamily='serif')
fig.text(0.16, 0.9, 
'''We observe that some ratings are only applicable to Movies. 
The most common for both Movies & TV Shows are TV-MA and TV-14.
'''
, fontsize=12, fontweight='light', fontfamily='serif')


fig.text(0.755,1,"Movie", fontweight="bold", fontfamily='serif', fontsize=15, color='#b20710')
fig.text(0.815,1,"|", fontweight="bold", fontfamily='serif', fontsize=15, color='black')
fig.text(0.825,1,"TV Show", fontweight="bold", fontfamily='serif', fontsize=15, color='#221f1f')

st.pyplot(fig)

st.markdown("<br><br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    color = ["#b20710", "#221f1f"]

    for i, mtv in enumerate(df['type'].value_counts().index):
        mtv_rel = df[df['type']==mtv]['year_added'].value_counts().sort_index()
        ax.plot(mtv_rel.index, mtv_rel, color=color[i], label=mtv)
        ax.fill_between(mtv_rel.index, 0, mtv_rel, color=color[i], alpha=0.9)
    
    ax.yaxis.tick_right()
    
    ax.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = .7)

    #ax.set_ylim(0, 50)
    #ax.legend(loc='upper left')
    for s in ['top', 'right','bottom','left']:
        ax.spines[s].set_visible(False)

    ax.grid(False)

    ax.set_xlim(2008,2020)
    plt.xticks(np.arange(2008, 2021, 1))

    fig.text(0.13, 0.85, 'Movies & TV Shows added over time', fontsize=15, fontweight='bold', fontfamily='serif')
    fig.text(0.13, 0.59, 
    '''    We see a slow start for Netflix over several years. 
    Things begin to pick up in 2015 and then there is a 
    rapid increase from 2016.

    It looks like content additions have slowed down in 2020, 
    likely due to the COVID-19 pandemic.
    '''

    , fontsize=12, fontweight='light', fontfamily='serif')


    fig.text(0.13,0.2,"Movie", fontweight="bold", fontfamily='serif', fontsize=15, color='#b20710')
    fig.text(0.19,0.2,"|", fontweight="bold", fontfamily='serif', fontsize=15, color='black')
    fig.text(0.2,0.2,"TV Show", fontweight="bold", fontfamily='serif', fontsize=15, color='#221f1f')

    ax.tick_params(axis=u'both', which=u'both',length=0)

    st.pyplot(fig)
with col2 :
    data_sub = df.groupby('type')['year_added'].value_counts().unstack().fillna(0).loc[['TV Show','Movie']].cumsum(axis=0).T

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    color = ["#b20710", "#221f1f"]

    for i, mtv in enumerate(df['type'].value_counts().index):
        mtv_rel = data_sub[mtv]
        ax.fill_between(mtv_rel.index, 0, mtv_rel, color=color[i], label=mtv,alpha=0.9)
    

    
    ax.yaxis.tick_right()
    
    ax.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = .7)

    #ax.set_ylim(0, 50)
    #ax.legend(loc='upper left')
    for s in ['top', 'right','bottom','left']:
        ax.spines[s].set_visible(False)

    ax.grid(False)

    ax.set_xlim(2008,2020)
    plt.xticks(np.arange(2008, 2021, 1))

    fig.text(0.13, 0.85, 'Movies & TV Shows added over time [Cumulative Total]', fontsize=15, fontweight='bold', fontfamily='serif')
    fig.text(0.13, 0.58, 
    '''Netflix peak global content amount was in 2019.

    It appears that Netflix has focused more attention
    on increasing Movie content that TV Shows. 
    Movies have increased much more dramatically
    than TV shows.
    '''

    , fontsize=12, fontweight='light', fontfamily='serif')



    fig.text(0.13,0.2,"Movie", fontweight="bold", fontfamily='serif', fontsize=15, color='#b20710')
    fig.text(0.19,0.2,"|", fontweight="bold", fontfamily='serif', fontsize=15, color='black')
    fig.text(0.2,0.2,"TV Show", fontweight="bold", fontfamily='serif', fontsize=15, color='#221f1f')

    ax.tick_params(axis=u'both', which=u'both',length=0)

    st.pyplot(fig)

st.markdown("<br><br>", unsafe_allow_html=True)
#Month-by-Month
col1, col2 = st.columns(2)
with col1 :
    month_order = ['January',
    'February',
    'March',
    'April',
    'May',
    'June',
    'July',
    'August',
    'September',
    'October',
    'November',
    'December']

    df['month_name_added'] = pd.Categorical(df['month_name_added'], categories=month_order, ordered=True)

    data_sub = df.groupby('type')['month_name_added'].value_counts().unstack().fillna(0).loc[['TV Show','Movie']].cumsum(axis=0).T

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    color = ["#b20710", "#221f1f"]

    for i, mtv in enumerate(df['type'].value_counts().index):
        mtv_rel = data_sub[mtv]
        ax.fill_between(mtv_rel.index, 0, mtv_rel, color=color[i], label=mtv,alpha=0.9)
        

        
    ax.yaxis.tick_right()
        
    ax.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = .4)

    #ax.set_ylim(0, 50)
    #ax.legend(loc='upper left')
    for s in ['top', 'right','bottom','left']:
        ax.spines[s].set_visible(False)

    ax.grid(False)
    ax.set_xticklabels(data_sub.index, fontfamily='serif', rotation=0)
    ax.margins(x=0) # remove white spaces next to margins

    #ax.set_xlim(2008,2020)
    #plt.xticks(np.arange(2008, 2021, 1))

    fig.text(0.13, 0.95, 'Content added by month [Cumulative Total]', fontsize=15, fontweight='bold', fontfamily='serif')
    fig.text(0.13, 0.905, 
    "The end & beginnings of each year seem to be Netflix's preference for adding content."

    , fontsize=12, fontweight='light', fontfamily='serif')



    fig.text(0.13,0.855,"Movie", fontweight="bold", fontfamily='serif', fontsize=15, color='#b20710')
    fig.text(0.19,0.855,"|", fontweight="bold", fontfamily='serif', fontsize=15, color='black')
    fig.text(0.2,0.855,"TV Show", fontweight="bold", fontfamily='serif', fontsize=15, color='#221f1f')


    ax.tick_params(axis=u'both', which=u'both',length=0)

    st.pyplot(fig)
with col2 :
    import matplotlib.colors

    df_copy=df
    data = df_copy.groupby('first_country')[['count']].sum().sort_values(by='count',ascending=False).reset_index()[:10]
    data = data['first_country']

    df_heatmap = df_copy.loc[df_copy['first_country'].isin(data)]
    df_heatmap = pd.crosstab(df_heatmap['first_country'],df_heatmap['target_ages'],normalize = "index").T

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    country_order2 = ['USA', 'India', 'UK', 'Canada', 'Japan', 'France', 'S. Korea', 'Spain',
        'Mexico']

    age_order = ['Kids','Older Kids','Teens','Adults']
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#221f1f', '#b20710','#f5f5f1'])

    sns.heatmap(df_heatmap.loc[age_order,country_order2],cmap=cmap,square=True, linewidth=2.5,cbar=False,
                annot=True,fmt='1.0%',vmax=.6,vmin=0.05,ax=ax,annot_kws={"fontsize":12})

    ax.spines['top'].set_visible(True)


    fig.text(.99, .77, 'Target ages proportion of total content by country', fontweight='bold', fontfamily='serif', fontsize=15,ha='right')   
    fig.text(0.99, .74, 'Here we see interesting differences between countries. Most shows in India are targeted to teens, for instance.',ha='right', fontsize=12,fontfamily='serif') 

    ax.set_yticklabels(ax.get_yticklabels(), fontfamily='serif', rotation = 0, fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), fontfamily='serif', rotation=90, fontsize=11)

    ax.set_ylabel('')    
    ax.set_xlabel('')
    ax.tick_params(axis=u'both', which=u'both',length=0)
    plt.tight_layout()

    st.pyplot(fig)



st.markdown("<br><br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1 :
    # Genres
    from sklearn.preprocessing import MultiLabelBinarizer 

    import matplotlib.colors


    # Custom colour map based on Netflix palette
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#221f1f', '#b20710','#f5f5f1'])
    def genre_heatmap(df_2, title):
        df_2=df
        df_2['genre_2'] = df_2['genre']
        Types = []
        for i in df_2['genre_2']: Types += i
        Types = set(Types)
        print("There are {} types in the Netflix {} Dataset".format(len(Types),title))    
        test = df['genre_2']
        mlb = MultiLabelBinarizer()
        res = pd.DataFrame(mlb.fit_transform(test), columns=mlb.classes_, index=test.index)
        corr = res.corr()
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        fig, ax = plt.subplots(figsize=(6,5))
        fig.text(.45,.88,'Genre correlation', fontfamily='serif',fontweight='bold',fontsize=12)
        fig.text(.72,.67,
                '''
                It is interesting that Independant Movies
                tend to be Dramas. 
                
                Another observation is that 
                Internatinal Movies are rarely
                in the Children's genre.
                ''', fontfamily='serif',fontsize=5,ha='right')
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.5, vmin=-.5, center=0, square=True, linewidths=1.5)
        ax.tick_params(axis='x',labelsize=5)
        ax.tick_params(axis='y',labelsize=5)
        return fig
    fig = genre_heatmap(data_sub, 'Subset')
            
    st.pyplot(fig)

with col2 :
    data_sub2 = data_sub
    data_sub2['Value'] = data_sub2['Movie'] + data_sub2['TV Show']
    data_sub2 = data_sub2.reset_index()

    df_polar = data_sub2.sort_values(by='month_name_added',ascending=False)


    color_map = ['#221f1f' for _ in range(12)]
    color_map[0] = color_map[11] =  '#b20710' # color highlight


    # initialize the figure
    fig, ax = plt.subplots(figsize=(10,6),dpi=10,subplot_kw={'polar': True})
    #ax = plt.subplot(111, polar=True)
    plt.axis('off')

    # Constants = parameters controling the plot layout:
    upperLimit = 30
    lowerLimit = 1
    labelPadding = 10

    # Compute max and min in the dataset
    max = df_polar['Value'].max()

    # Let's compute heights: they are a conversion of each item value in those new coordinates
    # In our example, 0 in the dataset will be converted to the lowerLimit (10)
    # The maximum will be converted to the upperLimit (100)
    slope = (max - lowerLimit) / max
    heights = slope * df_polar.Value + lowerLimit

    # Compute the width of each bar. In total we have 2*Pi = 360°
    width = 2*np.pi / len(df_polar.index)

    # Compute the angle each bar is centered on:
    indexes = list(range(1, len(df_polar.index)+1))
    angles = [element * width for element in indexes]

    # Draw bars
    bars = ax.bar(
        x=angles, 
        height=heights, 
        width=width, 
        bottom=lowerLimit,
        linewidth=2, 
        edgecolor="white",
        color=color_map,alpha=0.8
    )

    # Add labels
    for bar, angle, height, label in zip(bars,angles, heights, df_polar["month_name_added"]):

        # Labels are rotated. Rotation must be specified in degrees :(
        rotation = np.rad2deg(angle)

        # Flip some labels upside down
        alignment = ""
        if angle >= np.pi/2 and angle < 3*np.pi/2:
            alignment = "right"
            rotation = rotation + 180
        else: 
            alignment = "left"

        # Finally add the labels
        ax.text(
            x=angle, 
            y=lowerLimit + bar.get_height() + labelPadding, 
            s=label, 
            ha=alignment, fontsize=5,fontfamily='serif',
            va='center', 
            rotation=rotation, 
            rotation_mode="anchor") 
        
    st.pyplot(fig)

st.markdown("<br><br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1 :
    # Data

    df_movies=df
    #df_tv

    ### Relevant groupings

    data = df_movies.groupby('first_country')[['count']].sum().sort_values(by='count',ascending=False).reset_index()[:10]
    data = data['first_country']
    df_loli = df_movies.loc[df_movies['first_country'].isin(data)]

    loli = df_loli.groupby('first_country')[['release_year','year_added']].mean().round()

    # Reorder it following the values of the first value
    ordered_df = loli.sort_values(by='release_year')

    ordered_df_rev = loli.sort_values(by='release_year',ascending=False)

    my_range=range(1,len(loli.index)+1)


    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    fig.text(0.13, 0.9, 'How old are the movies? [Average]', fontsize=15, fontweight='bold', fontfamily='serif')
    plt.hlines(y=my_range, xmin=ordered_df['release_year'], xmax=ordered_df['year_added'], color='grey', alpha=0.4)
    plt.scatter(ordered_df['release_year'], my_range, color='#221f1f',s=100, alpha=0.9, label='Average release date')
    plt.scatter(ordered_df['year_added'], my_range, color='#b20710',s=100, alpha=0.9 , label='Average added date')
    #plt.legend()

    for s in ['top', 'left', 'right', 'bottom']:
        ax.spines[s].set_visible(False)
        

    # Removes the tick marks but keeps the labels
    ax.tick_params(axis=u'both', which=u'both',length=0)
    # Move Y axis to the right side
    ax.yaxis.tick_right()

    plt.yticks(my_range, ordered_df.index)
    plt.yticks(fontname = "serif",fontsize=12)

    # Custome legend
    fig.text(0.19,0.175,"Released", fontweight="bold", fontfamily='serif', fontsize=12, color='#221f1f')
    fig.text(0.76,0.175,"Added", fontweight="bold", fontfamily='serif', fontsize=12, color='#b20710')


    fig.text(0.10, 0.56, 
    '''
    The average gap between when 
    content is released, and when it
    is then added on Netflix varies
    by country. 

    In Spain, Netflix appears to be 
    dominated by newer movies 
    whereas Egypt & India have
    an older average movie.
    '''

    , fontsize=10, fontweight='light', fontfamily='serif')

    st.pyplot(fig)

with col2 :

    df_tv=df
    data = df_tv.groupby('first_country')[['count']].sum().sort_values(by='count',ascending=False).reset_index()[:10]
    data = data['first_country']
    df_loli = df_tv.loc[df_tv['first_country'].isin(data)]

    loli = df_loli.groupby('first_country')[['release_year','year_added']].mean().round()


    # Reorder it following the values of the first value:
    ordered_df = loli.sort_values(by='release_year')

    ordered_df_rev = loli.sort_values(by='release_year',ascending=False)

    my_range=range(1,len(loli.index)+1)


    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    fig.text(0.13, 0.9, 'How old are the TV shows? [Average]', fontsize=15, fontweight='bold', fontfamily='serif')
    plt.hlines(y=my_range, xmin=ordered_df['release_year'], xmax=ordered_df['year_added'], color='grey', alpha=0.4)
    plt.scatter(ordered_df['release_year'], my_range, color='#221f1f',s=100, alpha=0.9, label='Average release date')
    plt.scatter(ordered_df['year_added'], my_range, color='#b20710',s=100, alpha=0.9 , label='Average added date')
    #plt.legend()

    for s in ['top', 'left', 'right', 'bottom']:
        ax.spines[s].set_visible(False)
        
    ax.yaxis.tick_right()
    plt.yticks(my_range, ordered_df.index)
    plt.yticks(fontname = "serif",fontsize=12)


    fig.text(0.19,0.175,"Released", fontweight="bold", fontfamily='serif', fontsize=12, color='#221f1f')

    fig.text(0.47,0.175,"Added", fontweight="bold", fontfamily='serif', fontsize=12, color='#b20710')


    fig.text(0.10, 0.53, 
    '''
    The gap for TV shows seems
    more regular than for movies.

    This is likely due to subsequent
    series being released
    year-on-year.

    Spain seems to have
    the newest content
    overall.
    '''

    , fontsize=10, fontweight='light', fontfamily='serif')


    ax.tick_params(axis=u'both', which=u'both',length=0)

    st.pyplot(fig)


st.markdown("<br><br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1 :
    us_ind = df[(df['first_country'] == 'USA') | (df['first_country'] == 'India' )]

    data_sub = df.groupby('first_country')['year_added'].value_counts().unstack().fillna(0).loc[['USA','India']].cumsum(axis=0).T

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    color = ['#221f1f', '#b20710','#f5f5f1']

    for i, hs in enumerate(us_ind['first_country'].value_counts().index):
        hs_built = us_ind[us_ind['first_country']==hs]['year_added'].value_counts().sort_index()
        ax.plot(hs_built.index, hs_built, color=color[i], label=hs)
        #ax.fill_between(hs_built.index, 0, hs_built, color=color[i], alpha=0.4)
        ax.fill_between(hs_built.index, 0, hs_built, color=color[i], label=hs)
        

    ax.set_ylim(0, 1000)
    #ax.legend(loc='upper left')
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)

    ax.yaxis.tick_right()
        
    ax.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = .4)

    #ax.set_ylim(0, 50)
    #ax.legend(loc='upper left')
    for s in ['top', 'right','bottom','left']:
        ax.spines[s].set_visible(False)

    ax.grid(False)
    ax.set_xticklabels(data_sub.index, fontfamily='serif', rotation=0)
    ax.margins(x=0) # remove white spaces next to margins

    ax.set_xlim(2008,2020)
    plt.xticks(np.arange(2008, 2021, 1))

    fig.text(0.13, 0.85, 'USA vs. India: When was content added?', fontsize=15, fontweight='bold', fontfamily='serif')
    fig.text(0.13, 0.58, 
    '''
    We know from our work above that Netflix is dominated by the USA & India.
    It would also be reasonable to assume that, since Netflix is an American
    compnany, Netflix increased content first in the USA, before 
    other nations. 

    That is exactly what we see here; a slow and then rapid
    increase in content for the USA, followed by Netflix 
    being launched to the Indian market in 2016.'''

    , fontsize=12, fontweight='light', fontfamily='serif')



    fig.text(0.13,0.15,"India", fontweight="bold", fontfamily='serif', fontsize=15, color='#b20710')
    fig.text(0.188,0.15,"|", fontweight="bold", fontfamily='serif', fontsize=15, color='black')
    fig.text(0.198,0.15,"USA", fontweight="bold", fontfamily='serif', fontsize=15, color='#221f1f')


    ax.tick_params(axis=u'both', which=u'both',length=0)

    st.pyplot(fig)

with col2 :
    us_ind = df[(df['first_country'] == 'USA') | (df['first_country'] == 'India' )]

    data_sub = df.groupby('first_country')['year_added'].value_counts().unstack().fillna(0).loc[['USA','India']].cumsum(axis=0).T
    data_sub.insert(0, "base", np.zeros(len(data_sub)))

    data_sub = data_sub.add(-us_ind['year_added'].value_counts()/2, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    color = ['#b20710','#221f1f'][::-1]
    hs_list = data_sub.columns
    hs_built = data_sub[hs]

    for i, hs in enumerate(hs_list):
        if i == 0 : continue
        ax.fill_between(hs_built.index, data_sub.iloc[:,i-1], data_sub.iloc[:,i], color=color[i-1])
        
    for s in ['top', 'right', 'bottom', 'left']:
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    ax.set_yticks([])
    #ax.legend(loc='upper left')
    ax.grid(False)

    fig.text(0.16, 0.76, 'USA vs. India: Stream graph of new content added', fontsize=15, fontweight='bold', fontfamily='serif')
    fig.text(0.16, 0.575, 
    '''
    Seeing the data displayed like this helps 
    us to realise just how much content is added in the USA.
    Remember, India has the second largest amount of
    content yet is dwarfed by the USA.'''

    , fontsize=12, fontweight='light', fontfamily='serif')

    fig.text(0.16,0.41,"India", fontweight="bold", fontfamily='serif', fontsize=15, color='#b20710')
    fig.text(0.208,0.41,"|", fontweight="bold", fontfamily='serif', fontsize=15, color='black')
    fig.text(0.218,0.41,"USA", fontweight="bold", fontfamily='serif', fontsize=15, color='#221f1f')


    ax.tick_params(axis=u'y', which=u'both',length=0)

    st.pyplot(fig)

st.markdown("<br><br>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    
    from wordcloud import WordCloud
    import random
    from PIL import Image
    import matplotlib

    # Custom colour map based on Netflix palette
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#221f1f', '#b20710'])

    text = str(list(df['title'])).replace(',', '').replace('[', '').replace("'", '').replace(']', '').replace('.', '')

    mask = np.array(Image.open(r'project_sample\icon\netflix-logo.png'))

    wordcloud = WordCloud(background_color = 'white', width = 50,  height = 50,colormap=cmap, max_words = 150, mask = mask).generate(text)

    fig, ax = plt.subplots( figsize=(6,5))
    plt.imshow(wordcloud, interpolation = 'bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)

    st.pyplot(fig)
