# Encoding Random Forests with SAT for explainability
## Advanced Artificial Intelligence course final project
The purpose of this project is to encode random forest models with SAT as explained in [*On Explaining Random Forests with SAT*](https://www.ijcai.org/proceedings/2021/0356.pdf) paper by Yacine Izza and Joao Marques-Silva with the final aim of explain model predictions.
### Contents

#### docs
* **documentation** : examples of utilization of the implemented functions


#### src 
* **data** : files related to datasets on which random forest model has been trained and tested
* **lib** : libraries used or created
* *encode_rf_utils.py* : implemented function to encode a Scikit learn random forest model with SAT
* *mock_model.py* : contains classes to instantiate a mock random forest model to be used in tests

* **model** : files related to the random forest model
* *create_random_forest.py* : python script to generate a random forest model on the Iris dataset as described in the reference paper

### References
To generate the random forest model [Scikit learn](https://scikit-learn.org/stable/) library has been used.\
To encode and manipulate boolean and pseudo-boolean formulas [PySAT](https://pysathq.github.io/) library has been used.\
The original work: [Yacine Izza, Joao Marques-Silva. 2021. On Explaining Random Forests with SAT]

@inproceedings{ims-ijcai21,\
  author       = {Yacine Izza and\
                  Jo{\~{a}}o Marques{-}Silva},\
  editor       = {Zhi{-}Hua Zhou},\
  title        = {On Explaining Random Forests with {SAT}},\
  booktitle    = {Proceedings of the Thirtieth International Joint Conference on Artificial\
                  Intelligence, {IJCAI} 2021, Virtual Event / Montreal, Canada, 19-27\
                  August 2021},\
  pages        = {2584--2591},\
  publisher    = {ijcai.org},\
  year         = {2021},\
  url          = {https://doi.org/10.24963/ijcai.2021/356},\
  doi          = {10.24963/ijcai.2021/356}\
}\

And the [github repo](https://github.com/yizza91/RFxpl)
