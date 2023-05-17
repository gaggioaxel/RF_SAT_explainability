# Encoding Random Forests with SAT for explainability
## Advanced Artificial Intelligence course final project
The purpose of this project is to encode random forest models with SAT as explained in [*On Explaining Random Forests with SAT*](https://www.ijcai.org/proceedings/2021/0356.pdf) paper by Yacine Izza and Joao Marques-Silva with the final aim of explain model predictions.
### Contents
* **data** : files related to datasets on which random forest model has been trained and tested
* **examples** : examples of utilization of the implemented functions
* **model** : files related to the random forest model
* *create_random_forest.py* : python script to generate a random forest model on the Iris dataset as described in the reference paper
* *encode_rf_utils.py* : implemented function to encode a Scikit learn random forest model with SAT
* *mock_model.py* : contains classes to instantiate a mock random forest model to be used in tests
### References
To generate the random forest model [Scikit learn](https://scikit-learn.org/stable/) library has been used.\
To encode and manipulate boolean and pseudo-boolean formulas [PySAT](https://pysathq.github.io/) library has been used.
