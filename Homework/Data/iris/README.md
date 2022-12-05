# iris
The famous [iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set)!

[_Source_][Source]

## Preview

|     | sepal_length| sepal_width| petal_length| petal_width| species        |
|:----|------------:|-----------:|------------:|-----------:|:---------------|
| 1   |          5.1|         3.5|          1.4|         0.2| Iris-setosa    |
| 2   |          4.9|         3.0|          1.4|         0.2| Iris-setosa    |
| 3   |          4.7|         3.2|          1.3|         0.2| Iris-setosa    |
| 4   |          4.6|         3.1|          1.5|         0.2| Iris-setosa    |
| 5   |          5.0|         3.6|          1.4|         0.2| Iris-setosa    |
| ... |          ...|         ...|          ...|         ...| ...            |
| 146 |          6.7|         3.0|          5.2|         2.3| Iris-virginica |
| 147 |          6.3|         2.5|          5.0|         1.9| Iris-virginica |
| 148 |          6.5|         3.0|          5.2|         2.0| Iris-virginica |
| 149 |          6.2|         3.4|          5.4|         2.3| Iris-virginica |
| 150 |          5.9|         3.0|          5.1|         1.8| Iris-virginica |


## How to load the data

### Python Pandas

```python
import pandas as pd
iris = pd.read_csv('Data/iris/iris.csv')
```

### R data.table

```r
library(data.table)
iris = fread('Data/iris/iris.csv')
```

[Source]: https://www.kaggle.com/datasets/uciml/iris?sort=published
