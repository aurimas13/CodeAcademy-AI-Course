# Netflix Titles
A tabular dataset of movies and tv shows ("titles") on Netflix. 

[_Source_][source]

## Preview

|      |show_id |type    |title      |director   |cast       |country    |date_added | release_year|rating |duration  |listed_in  |description |
|:-----|:-------|:-------|:----------|:----------|:----------|:----------|:----------|------------:|:------|:---------|:----------|:-----------|
| 1    |s1      |Movie   |Dick Jo... |Kirsten... |           |United ... |Septemb... |         2020|PG-13  |90 min    |Documen... |As her ...  |
| 2    |s2      |TV Show |Blood &... |           |Ama Qam... |South A... |Septemb... |         2021|TV-MA  |2 Seasons |Interna... |After c...  |
| 3    |s3      |TV Show |Ganglands  |Julien ... |Sami Bo... |           |Septemb... |         2021|TV-MA  |1 Season  |Crime T... |To prot...  |
| 4    |s4      |TV Show |Jailbir... |           |           |           |Septemb... |         2021|TV-MA  |1 Season  |Docuser... |Feuds, ...  |
| 5    |s5      |TV Show |Kota Fa... |           |Mayur M... |India      |Septemb... |         2021|TV-MA  |2 Seasons |Interna... |In a ci...  |
| ...  |...     |...     |...        |...        |...        |...        |...        |         ... |...    |...       |...        |...         |
| 8803 |s8803   |Movie   |Zodiac     |David F... |Mark Ru... |United ... |Novembe... |         2007|R      |158 min   |Cult Mo... |A polit...  |
| 8804 |s8804   |TV Show |Zombie ... |           |           |           |July 1,... |         2018|TV-Y7  |2 Seasons |Kids' T... |While l...  |
| 8805 |s8805   |Movie   |Zombieland |Ruben F... |Jesse E... |United ... |Novembe... |         2009|R      |88 min    |Comedie... |Looking...  |
| 8806 |s8806   |Movie   |Zoom       |Peter H... |Tim All... |United ... |January... |         2006|PG     |88 min    |Childre... |Dragged...  |
| 8807 |s8807   |Movie   |Zubaan     |Mozez S... |Vicky K... |India      |March 2... |         2015|TV-14  |111 min   |Dramas,... |A scrap...  |


## How to load the data

### Python Pandas

```python
import pandas as pd
iris = pd.read_csv('Data/netflix-titles/netflix-titles.csv')
```

### R data.table

```r
library(data.table)
iris = fread('Data/netflix-titles/netflix-titles.csv')
```

[source]: https://www.kaggle.com/datasets/shivamb/netflix-shows?select=netflix_titles.csv