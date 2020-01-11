
# 项目：有关电影数据库数据集的调查

## 目录
<ul>
<li><a href="#intro">简介</a></li>
<li><a href="#wrangling">数据整理</a></li>
<li><a href="#eda">探索性数据分析</a></li>
<li><a href="#conclusions">结论</a></li>
</ul>

<a id='intro'></a>
## 简介

> 本数据集中包含 1 万条电影信息，信息来源为“电影数据库”（TMDb，The Movie Database），包括“用户评分”、“票房"、“演职人员 (cast)”、“电影类别 (genres)”、”经通货膨胀调整后的预算和票房”等信息。

> 在本次报告中，我们将主要围绕票房进行探索和分析。具体报告如下问题：
> 1. 电影评分和票房是否存在正向关系
> 2. 考察不同年份中, 不同电影类型的发行情况.
> 3. 按总收益来看, 哪些描述电影的关键字出现频率最多
> 4. 哪些导演的电影的票房比较高？




```python
# 用这个框对你计划使用的所有数据包进行设置
#   导入语句。
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings
import seaborn as sns; sns.set(style="white", color_codes=True)
%matplotlib inline
```

<a id='wrangling'></a>
## 数据整理



### 常规属性


```python
# 加载数据并打印几行。进行这几项操作，来检查数据
#   类型，以及是否有缺失数据或错误数据的情况。
movie_data = pd.read_csv('tmdb-movies.csv')
```

> 载入数据


```python
movie_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>imdb_id</th>
      <th>popularity</th>
      <th>budget</th>
      <th>revenue</th>
      <th>original_title</th>
      <th>cast</th>
      <th>homepage</th>
      <th>director</th>
      <th>tagline</th>
      <th>...</th>
      <th>overview</th>
      <th>runtime</th>
      <th>genres</th>
      <th>production_companies</th>
      <th>release_date</th>
      <th>vote_count</th>
      <th>vote_average</th>
      <th>release_year</th>
      <th>budget_adj</th>
      <th>revenue_adj</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>135397</td>
      <td>tt0369610</td>
      <td>32.985763</td>
      <td>150000000</td>
      <td>1513528810</td>
      <td>Jurassic World</td>
      <td>Chris Pratt|Bryce Dallas Howard|Irrfan Khan|Vi...</td>
      <td>http://www.jurassicworld.com/</td>
      <td>Colin Trevorrow</td>
      <td>The park is open.</td>
      <td>...</td>
      <td>Twenty-two years after the events of Jurassic ...</td>
      <td>124</td>
      <td>Action|Adventure|Science Fiction|Thriller</td>
      <td>Universal Studios|Amblin Entertainment|Legenda...</td>
      <td>6/9/15</td>
      <td>5562</td>
      <td>6.5</td>
      <td>2015</td>
      <td>1.379999e+08</td>
      <td>1.392446e+09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>76341</td>
      <td>tt1392190</td>
      <td>28.419936</td>
      <td>150000000</td>
      <td>378436354</td>
      <td>Mad Max: Fury Road</td>
      <td>Tom Hardy|Charlize Theron|Hugh Keays-Byrne|Nic...</td>
      <td>http://www.madmaxmovie.com/</td>
      <td>George Miller</td>
      <td>What a Lovely Day.</td>
      <td>...</td>
      <td>An apocalyptic story set in the furthest reach...</td>
      <td>120</td>
      <td>Action|Adventure|Science Fiction|Thriller</td>
      <td>Village Roadshow Pictures|Kennedy Miller Produ...</td>
      <td>5/13/15</td>
      <td>6185</td>
      <td>7.1</td>
      <td>2015</td>
      <td>1.379999e+08</td>
      <td>3.481613e+08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>262500</td>
      <td>tt2908446</td>
      <td>13.112507</td>
      <td>110000000</td>
      <td>295238201</td>
      <td>Insurgent</td>
      <td>Shailene Woodley|Theo James|Kate Winslet|Ansel...</td>
      <td>http://www.thedivergentseries.movie/#insurgent</td>
      <td>Robert Schwentke</td>
      <td>One Choice Can Destroy You</td>
      <td>...</td>
      <td>Beatrice Prior must confront her inner demons ...</td>
      <td>119</td>
      <td>Adventure|Science Fiction|Thriller</td>
      <td>Summit Entertainment|Mandeville Films|Red Wago...</td>
      <td>3/18/15</td>
      <td>2480</td>
      <td>6.3</td>
      <td>2015</td>
      <td>1.012000e+08</td>
      <td>2.716190e+08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>140607</td>
      <td>tt2488496</td>
      <td>11.173104</td>
      <td>200000000</td>
      <td>2068178225</td>
      <td>Star Wars: The Force Awakens</td>
      <td>Harrison Ford|Mark Hamill|Carrie Fisher|Adam D...</td>
      <td>http://www.starwars.com/films/star-wars-episod...</td>
      <td>J.J. Abrams</td>
      <td>Every generation has a story.</td>
      <td>...</td>
      <td>Thirty years after defeating the Galactic Empi...</td>
      <td>136</td>
      <td>Action|Adventure|Science Fiction|Fantasy</td>
      <td>Lucasfilm|Truenorth Productions|Bad Robot</td>
      <td>12/15/15</td>
      <td>5292</td>
      <td>7.5</td>
      <td>2015</td>
      <td>1.839999e+08</td>
      <td>1.902723e+09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>168259</td>
      <td>tt2820852</td>
      <td>9.335014</td>
      <td>190000000</td>
      <td>1506249360</td>
      <td>Furious 7</td>
      <td>Vin Diesel|Paul Walker|Jason Statham|Michelle ...</td>
      <td>http://www.furious7.com/</td>
      <td>James Wan</td>
      <td>Vengeance Hits Home</td>
      <td>...</td>
      <td>Deckard Shaw seeks revenge against Dominic Tor...</td>
      <td>137</td>
      <td>Action|Crime|Thriller</td>
      <td>Universal Pictures|Original Film|Media Rights ...</td>
      <td>4/1/15</td>
      <td>2947</td>
      <td>7.3</td>
      <td>2015</td>
      <td>1.747999e+08</td>
      <td>1.385749e+09</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



> 直观的了解数据每列包含的信息


```python
movie_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10866 entries, 0 to 10865
    Data columns (total 21 columns):
    id                      10866 non-null int64
    imdb_id                 10856 non-null object
    popularity              10866 non-null float64
    budget                  10866 non-null int64
    revenue                 10866 non-null int64
    original_title          10866 non-null object
    cast                    10790 non-null object
    homepage                2936 non-null object
    director                10822 non-null object
    tagline                 8042 non-null object
    keywords                9373 non-null object
    overview                10862 non-null object
    runtime                 10866 non-null int64
    genres                  10843 non-null object
    production_companies    9836 non-null object
    release_date            10866 non-null object
    vote_count              10866 non-null int64
    vote_average            10866 non-null float64
    release_year            10866 non-null int64
    budget_adj              10866 non-null float64
    revenue_adj             10866 non-null float64
    dtypes: float64(4), int64(6), object(11)
    memory usage: 1.7+ MB


> 通过Dataframe的info()函数，了解数据整体结构以及整洁程度。我们需要探索的变量包括电影的受欢迎程度、电影的评分、电影的导演、电影的上映时间、经通货膨胀调整后的电影票房和电影预算等，以上变量从上述表格中看空值较少，信息较为完整。

### 针对探索问题进行数据清理


```python
# 在讨论数据结构和需要解决的任何问题之后，
#   在本部分的第二小部分进行这些清理步骤。

```


```python
movie_data_copy = movie_data
```

> 拷贝数据


```python
movie_data_copy.columns
```




    Index(['id', 'imdb_id', 'popularity', 'budget', 'revenue', 'original_title',
           'cast', 'homepage', 'director', 'tagline', 'keywords', 'overview',
           'runtime', 'genres', 'production_companies', 'release_date',
           'vote_count', 'vote_average', 'release_year', 'budget_adj',
           'revenue_adj'],
          dtype='object')




```python
movie_data_copy = movie_data_copy[['popularity','director','vote_average','budget_adj','revenue_adj']]
```


```python
movie_data_copy.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>popularity</th>
      <th>director</th>
      <th>vote_average</th>
      <th>budget_adj</th>
      <th>revenue_adj</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>32.985763</td>
      <td>Colin Trevorrow</td>
      <td>6.5</td>
      <td>1.379999e+08</td>
      <td>1.392446e+09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>28.419936</td>
      <td>George Miller</td>
      <td>7.1</td>
      <td>1.379999e+08</td>
      <td>3.481613e+08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.112507</td>
      <td>Robert Schwentke</td>
      <td>6.3</td>
      <td>1.012000e+08</td>
      <td>2.716190e+08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.173104</td>
      <td>J.J. Abrams</td>
      <td>7.5</td>
      <td>1.839999e+08</td>
      <td>1.902723e+09</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.335014</td>
      <td>James Wan</td>
      <td>7.3</td>
      <td>1.747999e+08</td>
      <td>1.385749e+09</td>
    </tr>
  </tbody>
</table>
</div>



> 选取需要的列


```python
movie_data_copy.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10866 entries, 0 to 10865
    Data columns (total 5 columns):
    popularity      10866 non-null float64
    director        10822 non-null object
    vote_average    10866 non-null float64
    budget_adj      10866 non-null float64
    revenue_adj     10866 non-null float64
    dtypes: float64(4), object(1)
    memory usage: 424.5+ KB



```python
movie_data_copy = movie_data_copy.dropna()
```


```python
movie_data_copy.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 10822 entries, 0 to 10865
    Data columns (total 5 columns):
    popularity      10822 non-null float64
    director        10822 non-null object
    vote_average    10822 non-null float64
    budget_adj      10822 non-null float64
    revenue_adj     10822 non-null float64
    dtypes: float64(4), object(1)
    memory usage: 507.3+ KB


> 删除空值


```python
movie_data_copy = movie_data_copy[~movie_data_copy['revenue_adj'].isin([0])]
```


```python
movie_data_copy = movie_data_copy[~movie_data_copy['budget_adj'].isin([0])]
```


```python
movie_data_copy.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 3854 entries, 0 to 10848
    Data columns (total 5 columns):
    popularity      3854 non-null float64
    director        3854 non-null object
    vote_average    3854 non-null float64
    budget_adj      3854 non-null float64
    revenue_adj     3854 non-null float64
    dtypes: float64(4), object(1)
    memory usage: 180.7+ KB


> 删除在预算和票房中为0的无效数据

<a id='eda'></a>
## 探索性数据分析

> **提示**在你完成数据整理和清理之后，现在可以进行探索性数据分析了。你需要计算统计值，创建可视化图表，解决你在一开始的简介部分中提出的研究问题。我们推荐你采用系统化方法。一次只探索一个变量，然后探索变量之间的关系。

### 研究问题 1（电影评分和票房是否存在正向关系 ）

#### 图例1


```python
warnings.simplefilter("ignore")
j = sns.jointplot(data=movie_data_copy, y='revenue_adj',x='vote_average', kind="reg", size=8, space=0.5)
j.annotate(stats.pearsonr)
plt.title('Point diagram of box office and movie score',fontsize=15)
plt.show()
```


![png](output_27_0.png)


> 根据票房与评分的散点图可以看出票房与评分呈现出右偏的分布，评分高的影评出现高票房的概率更高。同时在上图中可以看出两者的线性关系，皮尔逊相关系数是0.27.

#### 图例2


```python
plt.figure(figsize=(10,6))
plt.title("The relation between vote and revenue - overview")
plt.scatter(movie_data_copy['vote_average'],movie_data_copy['revenue_adj'],marker='o',alpha = 1/3)
plt.savefig('test.png')
```


![png](output_30_0.png)


> 根据票房与评分的散点图可以看出票房与评分呈现出右偏的分布，评分高的影评出现高票房的概率更高。
> 但由于票房与评分皆为离散数据，出现重合的情况较多。故下图中，将根据票房分成四个截图，呈现更加细致的票房与评分的关系。

### 研究问题 2（考察不同年份中, 不同电影类型的发行情况.）


```python
%config InlineBackend.figure_format = 'retina'
plt.style.use('dark_background')
```


```python
# 拆分
df_genres = movie_data.drop('genres', axis=1).join(movie_data['genres'].str.split('|', expand=True).stack().reset_index(level=1, drop=True).rename('genres'))
target_data = df_genres.groupby('genres')['release_year'].value_counts().unstack().fillna(0)
target_data['sort_val'] = target_data.sum(axis=1)
target_data = target_data.sort_values('sort_val', axis=0, ascending=False).drop('sort_val', axis=1)
target_data.T.plot(colormap='gist_rainbow', figsize=(24, 8), grid=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1ce77940>




![png](output_34_1.png)


> From graph above, we can see the amount of genres of drama, comedy and thriller is increasing with time rapidly. 

### 研究问题 3（电影的预算和票房是否存在正向关系）


```python
from wordcloud import WordCloud
%config InlineBackend.figure_format = 'retina'
```


```python
# organize data
factor = 'revenue_adj'
kw_expand = movie_data['keywords'].str.split('|', expand=True).stack().reset_index(level=1, drop=True).rename('keywords')
df_kw_rev = movie_data[[factor]].join(kw_expand)
word_dict = df_kw_rev.groupby('keywords')[factor].sum().apply(lambda x: np.exp(x/1e11)).to_dict()

```


```python
# create wordcloud
params = {'mode': 'RGBA', 
          'background_color': 'rgba(255, 255, 255, 0)', 
          'colormap': 'Spectral'}
wordcloud = WordCloud(width=1200, height=800, **params)
wordcloud.generate_from_frequencies(word_dict)
```




    <wordcloud.wordcloud.WordCloud at 0x1a1ca68978>




```python
plt.figure(figsize=(8, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
```




    (-0.5, 1199.5, 799.5, -0.5)




![png](output_40_1.png)


 > From the graph, we can see “based on novel" is the most frequent keyword.

### 研究问题 4 (哪些导演的电影的票房比较高？)


```python
movie_data_director= movie_data_copy.groupby(['director'])['revenue_adj'].mean().reset_index().sort_values(['revenue_adj'],axis = 0,ascending = False)[0:9]

```


```python
plt.figure(figsize=(15,6))
plt.barh(movie_data_director['director'],movie_data_director['revenue_adj'])  
plt.show()  
```


![png](output_44_0.png)


> 上图中列示了所有电影的平均票房最高的导演，其中Irwin Winkler 执导的电影平均票房最高。

<a id='conclusions'></a>
## 结论

> 综上，主要围绕票房进行了数据性探索，得出的初步结论包括票房同电影评分、电影受欢迎程度以及预算呈现一定的正向关系。同时所有导演中，Irwin Winkler是平均票房最高的导演。
> 同时，上述分析存在一定的局限性：
> 1. 上述数据期限为1960 - 2015年的数据，而近年来电影市场非常较快，尤其是中国市场。故数据中缺少近年来的数据，以及缺少中国的影片的信息，可能导致整体分析存在不完整的情况。
> 2. 上述数据分析主要围绕了电影票房（因变量），电影的受欢迎程度、电影的评分、电影的导演、经通货膨胀调整后电影预算（若干自变量）进行分析，同时未考虑上述自变量之间可能存在的相关性。基于上述范围和假设，虽然本次探索分析只用了整个数据集的子集进行分析，但是使用的子集可以支撑得出的结论。
> 3. 本次分析对于异常值以及空值进行了全部删除，以免影响数据的准确性。但是，同时也导致了数据减少的情况，可能对于最终的结论产生一定的影响。
