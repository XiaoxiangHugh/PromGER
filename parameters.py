# In this file, users can modify the parameters used in training model.
# Users can also add a new dataset, by adding a new item to the dataset_parameters_dict.
# The form of items are {"dataset"_ name":parameters}

dataset_parameters_dict = {"Hs": {"features": ["NCP-ND", "EIIP", "BPB"],
                                  "neighbor_num": 40,
                                  "embeddings": {'GraRep': {'dimensions': 4, 'iteration': 40},
                                                 'Node2Vec': {'dimensions': 32},
                                                 'SocioDim': {'dimensions': 4}},
                                  "catboost": {"class_weights": [1, 2]}},
                           "Rn": {"features": ["NCP-ND", "EIIP", "BPB"],
                                  "neighbor_num": 350,
                                  "embeddings": {'GraRep': {'dimensions': 8, 'iteration': 40},
                                                 'Node2Vec': {'dimensions': 8},
                                                 'SocioDim': {'dimensions': 4}},
                                  "catboost": {"class_weights": [1, 2]}},
                           }
