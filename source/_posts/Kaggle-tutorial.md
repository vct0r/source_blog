title: Kaggle尝鲜：探索泰坦尼克号事故幸存情况分析
author: vvvict0r
tags:
  - Data Analysis
categories:
  - Data
date: 2017-12-18 21:43:00
---
Kaggle 入门：探索泰坦尼克号事故幸存情况分析

[原文地址](https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic)

作者：梅根·丽思达(Megan L. Risdal)

<!-- more --> 

---

文章目录：
+ 1 介绍
 - 1.1 载入并检查数据
+ 2 特征工程
 - 2.1 名字分析
 - 2.2 家庭存活情况分析
 - 2.3 处理更多变量
+ 3 缺失数据处理
 - 3.1 观测值插补
 - 3.2 预测性插补
 - 3.3 特征工程：第二阶段
+ 4 预测
 - 4.1 从源数据分离出训练数据和测试数据
 - 4.2 建立模型
 - 4.3 变量重要性分析
 - 4.4 开始预测！
+ 5 结论

---

## 1 介绍
这是我第一次尝试使用 Kaggle 脚本。经过一段时间在 Kaggle 上的浏览和阅读其他用户写的脚本后，我决定对泰坦尼克号事件情况数据集进行分析。在此期间我也将生成一些数据可视化图片。这个实例中我用来预测幸存者的模型是基于随机森林(`randomForest`)方法建立的。我才刚入门机器学习这个领域，还有很多东西要学，欢迎大家在评论区进行反馈！

在我脚本中主要包含三个部分：
* 特征工程
* 缺失数据插补
* 预测！

### 1.1 载入并检查数据

```
# 载入分析所用的包

# 这些是用于数据可视化的包
library('ggplot2')
library('ggthemes')
library('scales')

library('dplyr') # 数据处理包
library('mice') # 数据插补包
library('randomForest') # 随机森林分类算法
```

载入需要的包后，我们要读入数据：
```
train <- read.csv('../input/train.csv', stringsAsFactors = F)
test  <- read.csv('../input/test.csv', stringsAsFactors = F)

full  <- bind_rows(train, test) # 结合训练数据和测试数据

# 检查数据
str(full)
```

代码运行得出的结果：
```
## 'data.frame':    1309 obs. of  12 variables:
##  $ PassengerId: int  1 2 3 4 5 6 7 8 9 10 ...
##  $ Survived   : int  0 1 1 1 0 0 0 0 1 1 ...
##  $ Pclass     : int  3 1 3 1 3 3 1 3 3 2 ...
##  $ Name       : chr  "Braund, Mr. Owen Harris" "Cumings, Mrs. John Bradley (Florence Briggs Thayer)" "Heikkinen, Miss. Laina" "Futrelle, Mrs. Jacques Heath (Lily May Peel)" ...
##  $ Sex        : chr  "male" "female" "female" "female" ...
##  $ Age        : num  22 38 26 35 35 NA 54 2 27 14 ...
##  $ SibSp      : int  1 1 0 1 0 0 0 3 0 1 ...
##  $ Parch      : int  0 0 0 0 0 0 0 1 2 0 ...
##  $ Ticket     : chr  "A/5 21171" "PC 17599" "STON/O2. 3101282" "113803" ...
##  $ Fare       : num  7.25 71.28 7.92 53.1 8.05 ...
##  $ Cabin      : chr  "" "C85" "" "C123" ...
##  $ Embarked   : chr  "S" "C" "S" "S" ...
```

目前我们大概了解了所要分析的变量，它们的类型，以及一部分观测值。我们发现我们将要分析12个变量的1309个观测值。由于我们无法从一些变量名完全看出其意思，以下列表将帮助我们理解各个变量名所带有的含义：
|变量名|说明|
|---|---|
|Survived|幸存(1)与罹难(0)|
|Pclass|乘客等级|
|Name|乘客名字|
|Sex|乘客性别|
|Age|乘客年龄|
|SibSp|该乘客在船上的兄弟姐妹和配偶数量|
|Parch|该乘客在船上的双亲和儿女数量|
|Ticket|船票编号|
|Fare|船票价格|
|Cabin|舱位|
|Embarked|登船港口|

## 2 特征工程

### 2.1 名字分析

我首先注意到的变量是**乘客名字**，因为我们可以将这个变量切片出更多变量来用于帮助我们的预测，或是用于创建新的变量。比如，乘客名字中包含了**乘客头衔**，我们还可以利用**姓氏**来代表其家庭。现在就开始**特征工程**分析吧！

```
# 从乘客名字中分离出头衔
full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)

# 根据性别显示头衔
table(full$Sex, full$Title)
```

运行结果：
```
##         
##          Capt Col Don Dona  Dr Jonkheer Lady Major Master Miss Mlle Mme
##   female    0   0   0    1   1        0    1     0      0  260    2   1
##   male      1   4   1    0   7        1    0     2     61    0    0   0
##         
##           Mr Mrs  Ms Rev Sir the Countess
##   female   0 197   2   0   0            1
##   male   757   0   0   8   1            0
```

处理特别头衔：
```
# 将数量较少的头衔归类于“稀有头衔”(rare_title)
rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

# 对 Mlle, Ms, Mme 这类头衔重新归类，因为这类头衔意思重复
full$Title[full$Title == 'Mlle']        <- 'Miss' 
full$Title[full$Title == 'Ms']          <- 'Miss'
full$Title[full$Title == 'Mme']         <- 'Mrs' 
full$Title[full$Title %in% rare_title]  <- 'Rare Title'

# 再次根据性别显示头衔数量统计
table(full$Sex, full$Title)
```

运行结果：
```
##         
##          Master Miss  Mr Mrs Rare Title
##   female      0  264   0 198          4
##   male       61    0 757   0         25
```

接下来，从乘客名字中提取姓氏：
```
full$Surname <- sapply(full$Name,  
                      function(x) strsplit(x, split = '[,.]')[[1]][1])
```

于是我们得出一共有875个不一样的姓氏。

### 2.2 家庭存活情况分析
现在乘客名字已经被处理成新的变量了，我们可以将其更进一步创建家庭变量。首先我们要基于兄弟姐妹、伴侣(也许有人会有多个伴侣)、双亲及儿女数量来创建**家庭规模**变量。

```
# 创建家庭规模变量，其规模包括乘客自己，所以最后有+1
full$Fsize <- full$SibSp + full$Parch + 1

# 创建家庭变量
full$Family <- paste(full$Surname, full$Fsize, sep='_')
```

家庭规模变量是怎样的呢？为了更好了解其对幸存情况的影响，我们先根据训练数据中的这个变量来绘图：
```
# 使用 ggplot2 来创建家庭规模和幸存情况之间的关系图
ggplot(full[1:891,], aes(x = Fsize, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge') +
  scale_x_continuous(breaks=c(1:11)) +
  labs(x = 'Family Size') +
  theme_few()
```

?[img](https://www.kaggle.io/svf/924638/c05c7b2409e224c760cdfb527a8dcfc4/__results___files/figure-html/unnamed-chunk-6-1.png)

啊哈。现在我们能看出孤身一人在船上的和家庭规模在四人以上在船上的人幸存率较低。我们可以将这个变量归为三个等级，因为大型家庭的数量相对较少。现在创建**离散家庭规模**变量。

```
# 将家庭规模变量离散化，分成个人、小型家庭、大型家庭三类
full$FsizeD[full$Fsize == 1] <- 'singleton'
full$FsizeD[full$Fsize < 5 & full$Fsize > 1] <- 'small'
full$FsizeD[full$Fsize > 4] <- 'large'

# 通过使用马赛克图(mosaic plot)来展示不同家庭规模幸存情况
mosaicplot(table(full$FsizeD, full$Survived), main='Family Size by Survival', shade=TRUE)
```

马赛克图再次证明了我们之前的观点，即孤身一人在船上的和家庭中有四人以上在船上的人幸存率较低，而小型家庭幸存率则更高。我打算对年龄变量进行分析，然而数据集中有263行的年龄变量是缺失值，所以我们必须先处理好缺失值才能分析年龄。

## 2.3 处理更多变量
还有哪些变量可以进行特征工程分析呢？也许**舱位号**也是很有价值的变量，因为舱位号中包含了甲板编号。接下来就研究一下。

```
# 可以看出，这项变量包含了大量的缺失值
full$Cabin[1:28]
```

运行结果：
```
##  [1] ""            "C85"         ""            "C123"        ""           
##  [6] ""            "E46"         ""            ""            ""           
## [11] "G6"          "C103"        ""            ""            ""           
## [16] ""            ""            ""            ""            ""           
## [21] ""            "D56"         ""            "A6"          ""           
## [26] ""            ""            "C23 C25 C27"
```

```
# 舱位号的首个字符便是甲板编号。例如：
strsplit(full$Cabin[2], NULL)[[1]]
```

运行结果：
```
## [1] "C" "8" "5"
```

接下来创建甲板编号变量，获取从A - F的甲板编号：
```
full$Deck<-factor(sapply(full$Cabin, function(x) strsplit(x, NULL)[[1]][1]))
```

这部分我们还可以做很多，比如研究带有多个房间的船舱(例如，row 28: “C23 C25 C27”)，但考虑到这项变量比较少，我们不会深入研究这个。

## 3 缺失数据处理

现在我们将要开始研究缺失数据，通过插补来对其进行处理。我们有很多方法处理缺失数据。考虑到该数据集的规模较小，我们不会将带有缺失数据的观测(即行)或是变量(即列)完全删除。我们可以根据数据分布情况用合适的值替换缺失数据，这类值包括均值、中位数或是众数，或者，我们也可以用预测值。我们将使用后两种方法处理缺失值，而我会根据可视化数据来做出决定。

### 3.1 合理值插补

62号乘客和830号乘客的登船港口是缺失的：
```
full[c(62, 830), 'Embarked']
```

运行结果：
```
## [1] "" ""
```

经过一番考虑，我推断当前数据中有两个变量和登船港口有关，这两个变量是乘客等级和票价：
```
# 排除带有缺失值的乘客ID
embark_fare <- full %>%
  filter(PassengerId != 62 & PassengerId != 830)

# 使用ggplot2来可视化登船港口情况，乘客等级和中位数票价
ggplot(embark_fare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +
  geom_boxplot() +
  geom_hline(aes(yintercept=80), 
    colour='red', linetype='dashed', lwd=2) +
  scale_y_continuous(labels=dollar_format()) +
  theme_few()
```

?[img](https://www.kaggle.io/svf/924638/c05c7b2409e224c760cdfb527a8dcfc4/__results___files/figure-html/unnamed-chunk-11-1.png)

从图中我们可以看出，在 Charbourg ('C') 登船的头等舱乘客的票价中位数大概是$80，符合我们的缺失登船港口乘客数据的情况。所以，我们可以将这两个乘客的登船港口缺失值替换为'C'。

```
full$Embarked[c(62, 830)] <- 'C'
```

我们继续处理缺失数据，第1044行的乘客信息的票价是缺失的。

```
full[1044, ]
```

运行结果：
```
##      PassengerId Survived Pclass               Name  Sex  Age SibSp Parch
## 1044        1044       NA      3 Storey, Mr. Thomas male 60.5     0     0
##      Ticket Fare Cabin Embarked Title Surname Fsize   Family    FsizeD
## 1044   3701   NA              S    Mr  Storey     1 Storey_1 singleton
##      Deck
## 1044 <NA>
```

这是名由 Southampton('S') 登船的三等舱乘客。根据相同船舱等级和登船港口，我们对其他乘客的票价情况进行可视化处理。

```
ggplot(full[full$Pclass == '3' & full$Embarked == 'S', ], 
  aes(x = Fare)) +
  geom_density(fill = '#99d6ff', alpha=0.4) + 
  geom_vline(aes(xintercept=median(Fare, na.rm=T)),
    colour='red', linetype='dashed', lwd=1) +
  scale_x_continuous(labels=dollar_format()) +
  theme_few()
```

?[img](https://www.kaggle.io/svf/924638/c05c7b2409e224c760cdfb527a8dcfc4/__results___files/figure-html/unnamed-chunk-14-1.png)

根据这个数据可视化图，由于他们的船舱等级和登船港口一致，我们可以将该缺失值替换为该图中中位数的数值，即$8.05。

```
full$Fare[1044] <- median(full[full$Pclass == '3' & full$Embarked == 'S', ]$Fare, na.rm = TRUE)
```

### 3.2 预测性插补

就如我们在之前所说，该数据集中有一些缺失的**年龄**数据。我们将在插补年龄的缺失数据时采用一些更有意思的方法。我们会创建根据其他变量预测年龄值的模型。

显示缺失年龄值的数量：
```
sum(is.na(full$Age))
```

运行结果：
```
## [1] 263
```

我们也可以使用`rpart`包(递归划分回归)来预测缺失年龄值，但在这里我会使用`mice`包。你可以在[这个链接](http://www.jstatsoft.org/article/view/v045i03/v45i03.pdf)里阅读到更多有关使用R中的链式方程来进行多重插补的内容。我们先要分解因子变量，然后使用`mice`包来插补缺失数据。

```
# 分解因子变量
factor_vars <- c('PassengerId','Pclass','Sex','Embarked',
                 'Title','Surname','Family','FsizeD')

full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))

# 设置随机种子
set.seed(129)

# 使用mice包进行插补，除去无用变量，这里使用了随机森林方法
mice_mod <- mice(full[, !names(full) %in% c('PassengerId','Name','Ticket','Cabin','Family','Surname','Survived')], method='rf') 
```

运行结果：
```
## 
##  iter imp variable
##   1   1  Age  Deck
##   1   2  Age  Deck
##   1   3  Age  Deck
##   1   4  Age  Deck
##   1   5  Age  Deck
##   2   1  Age  Deck
##   2   2  Age  Deck
##   2   3  Age  Deck
##   2   4  Age  Deck
##   2   5  Age  Deck
##   3   1  Age  Deck
##   3   2  Age  Deck
##   3   3  Age  Deck
##   3   4  Age  Deck
##   3   5  Age  Deck
##   4   1  Age  Deck
##   4   2  Age  Deck
##   4   3  Age  Deck
##   4   4  Age  Deck
##   4   5  Age  Deck
##   5   1  Age  Deck
##   5   2  Age  Deck
##   5   3  Age  Deck
##   5   4  Age  Deck
##   5   5  Age  Deck
```

保存输出结果：
```
mice_output <- complete(mice_mod)
```

现在对比一下原数据中乘客年龄分布和我们得出的结果，确保没出现太大偏差：
```
# 绘出年龄分布图
par(mfrow=c(1,2))
hist(full$Age, freq=F, main='Age: Original Data', 
  col='darkgreen', ylim=c(0,0.04))
hist(mice_output$Age, freq=F, main='Age: MICE Output', 
  col='lightgreen', ylim=c(0,0.04))
```

?[img](https://www.kaggle.io/svf/924638/c05c7b2409e224c760cdfb527a8dcfc4/__results___files/figure-html/unnamed-chunk-18-1.png)

看起来效果不错，接下来我们利用`mice`模型中的数据来替换原数据里的年龄值：
```
# 替换数据
full$Age <- mice_output$Age

# 显示新数据中缺失数据的情况
sum(is.na(full$Age))
```

运行结果：
```
## [1] 0
```

终于完成了对需要变量的所有缺失数据的处理！现在我们有了完整的年龄变量值，接下来我打算做些收尾工作。我们之后可以利用年龄值做更多的特征工程。

### 3.3 特征工程：第二阶段
现在我们获得了所有人的年龄，我们可以创建一些基于年龄生成的变量：**母亲**和**孩子**。判断孩子的条件是18岁以下，而母亲的条件是：1)是女性，2)超过18岁，3)至少有一个小孩，4)头衔不是"小姐"。

我们首先了解一下年龄和幸存情况的关系：
```
ggplot(full[1:891,], aes(Age, fill = factor(Survived))) + 
  geom_histogram() + 
  # 我把性别也包括在内，因为我们知道性别也是一个很重要的因素
  facet_grid(.~Sex) + 
  theme_few()
```

?[img](https://www.kaggle.io/svf/924638/c05c7b2409e224c760cdfb527a8dcfc4/__results___files/figure-html/unnamed-chunk-20-1.png)

创建分辨孩子的列，并判断是孩子还是成人：
```
full$Child[full$Age < 18] <- 'Child'
full$Child[full$Age >= 18] <- 'Adult'

# 查看统计
table(full$Child, full$Survived)
```

运行结果：
```
##        
##           0   1
##   Adult 484 274
##   Child  65  68
```

看来孩子的幸存率确实会高一些，但即使是孩子也无法幸免于难。接下来，我们的特征工程还需要完成创建**母亲**变量。也许我们可以期待在泰坦尼克号上的母亲们能有更大的幸存几率。

```
# 加入母亲变量
full$Mother <- 'Not Mother'
full$Mother[full$Sex == 'female' & full$Parch > 0 & full$Age > 18 & full$Title != 'Miss'] <- 'Mother'

# 显示统计
table(full$Mother, full$Survived)
```

运行结果：
```
##             
##                0   1
##   Mother      16  39
##   Not Mother 533 303
```

我们需要的变量现在都已经处理好了，里面也没有缺失数据。谨慎起见，我再检查一下：
```
md.pattern(full)
```

运行结果：
```
## Warning in data.matrix(x): NAs introduced by coercion

## Warning in data.matrix(x): NAs introduced by coercion

## Warning in data.matrix(x): NAs introduced by coercion
```

```
##     PassengerId Pclass Sex Age SibSp Parch Fare Embarked Title Surname
## 150           1      1   1   1     1     1    1        1     1       1
##  61           1      1   1   1     1     1    1        1     1       1
##  54           1      1   1   1     1     1    1        1     1       1
## 511           1      1   1   1     1     1    1        1     1       1
##  30           1      1   1   1     1     1    1        1     1       1
## 235           1      1   1   1     1     1    1        1     1       1
## 176           1      1   1   1     1     1    1        1     1       1
##  92           1      1   1   1     1     1    1        1     1       1
##               0      0   0   0     0     0    0        0     0       0
##     Fsize Family FsizeD Child Mother Ticket Survived Deck Name Cabin     
## 150     1      1      1     1      1      1        1    1    0     0    2
##  61     1      1      1     1      1      1        0    1    0     0    3
##  54     1      1      1     1      1      0        1    1    0     0    3
## 511     1      1      1     1      1      1        1    0    0     0    3
##  30     1      1      1     1      1      0        0    1    0     0    4
## 235     1      1      1     1      1      1        0    0    0     0    4
## 176     1      1      1     1      1      0        1    0    0     0    4
##  92     1      1      1     1      1      0        0    0    0     0    5
##         0      0      0     0      0    352      418 1014 1309  1309 4402
```

太棒了！我们现在已经把所有泰坦尼克号数据集里的相关缺失值处理好了，其中用到的`mice`包也榜上了大忙。我们现在可以创建一些变量来帮我们建立预测幸存情况的模型。

## 4 预测

我们终于能根据我们处理好的数据开始预测在泰坦尼克号上的乘客幸存情况了。我们将使用随机森林`randomForest`分类算法来进行预测。

### 4.1 分离出训练数据和测试数据
我们的第一步是分离用于训练的数据和用于测试的数据。

```
train <- full[1:891,]
test <- full[892:1309,]
```

### 4.2 建立模型
然后对训练数据使用随机森林`randomForest`。

```
# 设置随机种子
set.seed(754)

# 建立模型(注意：不是所有变量都要使用)
rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 
                                            Fare + Embarked + Title + 
                                            FsizeD + Child + Mother,
                                            data = train)

# 显示模型误差
plot(rf_model, ylim=c(0,0.36))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)
```

?[img](https://www.kaggle.io/svf/924638/c05c7b2409e224c760cdfb527a8dcfc4/__results___files/figure-html/unnamed-chunk-24-1.png)

黑线表示总体误差率，大概在20%以下。红线和绿线分别表示死亡和幸存的误差率。现在我们就可以知道，我们预测死亡情况比预测幸存情况更为准确。那意味着什么呢？

### 4.3 变量重要性分析

我们来通过绘制平均精度下降图来看看变量重要性分布。

```
# 得到重要性
importance    <- importance(rf_model)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

# 创建基于重要性的排名变量
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

# 使用ggplot2来绘制相关变量重要性示意图
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
    y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
    hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() + 
  theme_few()
```

?[img](https://www.kaggle.io/svf/924638/c05c7b2409e224c760cdfb527a8dcfc4/__results___files/figure-html/unnamed-chunk-25-1.png)

哇，幸亏我们创建了头衔变量！头衔变量是这些变量中相关重要性最高的一个。同时我对乘客等级排在第五位感到惊奇，也许这是因为我们看了电影《泰坦尼克号》得到的偏见。

### 4.4 开始预测！
我们终于来到了最后一步：做出预测！但我们依旧可以重复前面的步骤来对预测进行调整，比如采用别的模型或是变量的不同组合，来达到更好的预测效果。

```
# 使用测试数据进行预测
prediction <- predict(rf_model, test)

# 用两个列把结果保存到数据框中：乘客ID(PassengerId)和预测幸存情况(Survived (prediction))
solution <- data.frame(PassengerID = test$PassengerId, Survived = prediction)

# 将结果写入文件中
write.csv(solution, file = 'rf_mod_Solution.csv', row.names = F)
```

## 5 结论
感谢你阅读我对Kaggle数据集的第一次尝试。我打算在这方面能做得更多，当然了，欢迎大家对这个新手的笔记进行评论和建议。

> 
注：本文由 vcvc 翻译自 [Megan L. Risdal. Exploring the Titanic Dataset](https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic)