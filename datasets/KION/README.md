# MTS Kion Implicit Contextualised Sequential Dataset for Movie Recommendation

## Dataset Description
This is an official repository of the Kion Movies Recommendation Dataset. 
The data was gathered from the users of [MTS Kion](https://kion.ru/home) video streaming platform from 13.03.2021 to 22.08.2022.

The public part of the  dataset includes 5,476,251 interactions of 962,179 users with 15,706 items. The dataset includes user_item interactions and their characteristics (such as temporal information, watch duration and watch percentage), as well as user demographics and rich movies meta-information.

*Private* part of the dataset contains movies that the users watched within one week following the period covered by the public dataset. It is not released to general public, however there is a public the sandbox, where the researchers can measure MAP@10 metric on the private part of the data. Sandbox is accessible by the address https://ods.ai/competitions/competition-recsys-21/leaderboard/public_sandbox. 

To make a submission, you need to use the [sample_submission.csv](https://github.com/irsafilo/KION_DATASET/blob/main/sample_submission.csv) file, and replace the sample item ids with the ids of the recommended items according to your recommendation model.


### The dataset consists of three parts: 
1. **Interactions.csv** - contains user-item implicit interactions, watch percentages, watch durations
2. **Users.cvs** - contains users demographics information (sex, age band, income level band, kids flag) 
3. **Items.cvs** - contains items meta-information (title, original title, year, genres, keywords, descriptions, countries, studios, actors, directors)

#### The users and items files have two versions: 

* **data_original** - original meta-information in Russian language
* **data_en** - english version of the metadata translated with Facebook FAIR’s WMT19 Ru->En machine translation model. 

## Comparison with MovieLens-25M and Netflix datasets

### Quantitative comparison:
|                              | **Netflix** |**Movielens-25M** | **Kion**           |
|------------------------------|-------------|------------------|--------------------|
| Users                        | 480,189     | 162,541          | 962,179            |
| Items                        | 17,770      | 59,047           | 15,706             |
| Interactions                 | 100,480,507 | 25,000,095       | 5,476,251          |
| Avg. Sequence Length         | 209.25      | 153.80           | 5.69               |
| Sparsity                     | 98.82%      | 99.73%           | 99.9%              |


### Qualitative comparison: 
| **Dataset Name**                       | **Netflix**.        |       **Movielens-25M**                  | **Kion**                        |
|----------------------------------------|---------------------|------------------------------------------|---------------------------------|
| Type                                   | Explicit (Ratings)  | Explicit (Rating)                        | Implicit (Interactions)         |
|         Interaction registration time. | After watching      | After watching                           | At watching                     |
|         Interaction features           | Date, Rating        | Date, Rating                             | Date, Duration, Watched Percent |
|         User features                  | None                | None                                     | Age, Income, Kids               |
|         Item features                  | Release Year, Title |Release Year, Title, Genres, Tags           | Content Type, Title, Original Title, Release Year, Genres, Countries, For Kids, Age Rating, Studios, Directors, Actors, Description, Keyword |

# Kion challenge
This dataset was used for the Kion challenge recommendation contest [ (Official website in Russian Language)](https://ods.ai/competitions/competition-recsys-21). 

This table contains results of the winners of the competition, measured on the private part of the dataset: 

| Position            | Name              | MAP@10 | Solution Type                  |
|---------------------|-------------------|--------|--------------------------------|
| 1                   | Oleg Lashinin     | 0.1221 | Neural and Non-Neural ensemble |
| 2                   | Aleksandr Petrov  | 0.1214 | Neural and Non-Neural ensemble |
| 3                   | Stepan Zimin      | 0.1213 | Non-Neural ensemble            |
| 4                   | Daria Tikhonovich | 0.1148 | Gradient Boosting Trees        |
| 5                   | Olga              | 0.1135 | Gradient Boosting Trees        |
| Popularity baseline |                   | 0.0910 |                                |

## Acknowledgements

[Igor Belkov](https://github.com/OzmundSedler), [Irina Elisova](https://github.com/ElisovaIra).

We would would like to acknowledge Kion challenge participants Oleg Lashinin,  Stepan Zimin, and Olga for providing descriptions of their Kion Challenge solutions, MTS Holding for providing the Kion dataset, ODS.ai international platform for hosting the competition. 

## Citations

If you use this dataset in your research, please cite our work: 

```
@article{petrov2022mts,
  title={MTS Kion Implicit Contextualised Sequential Dataset for Movie Recommendation},
  author={Aleksandr Petrov,  Ildar Safilo, Daria Tikhonovich and Dmitry Ignatov},
  year={2022}
}
```
