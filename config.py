import numpy as np
from ray.tune.variant_generator import grid_search

hyperparams = {
    "sd-innetaggr": {
        "repeat": 20,
        "resources": { "cpu": 4, "gpu": 1},
        "tune": {
            "state_description": True,
            "g_layers": [
                lambda spec: np.random.choice([128,256]), #lambda spec: np.random.choice([128,256]) if spec.config.aggregation_position!=1 else np.random.choice([1024,2048]),
                grid_search([1024, 2048]), #lambda spec: np.random.choice([128,256]) if spec.config.aggregation_position!=2 else np.random.choice([1024,2048]),
                lambda spec: np.random.choice([128,256]), #lambda spec: np.random.choice([128,256]) if spec.config.aggregation_position!=3 else np.random.choice([1024,2048]),
                lambda spec: np.random.choice([128,256]),
                lambda spec: np.random.choice([128,256]),
                grid_search([512, 1024])
            ],
            "aggregation_position": 2, #lambda spec: np.random.randint(1, 4),
            "question_injection_position": lambda spec: np.random.randint(3, 6), #lambda spec: spec.config.aggregation_position + np.random.randint(1, 3),
            "dropouts": {
                "4":lambda spec: np.random.uniform(high=0.5),
            },
                
            "lstm_hidden": 256,
            "lstm_word_emb": 32,
            "rl_in_size": 14
        }
    }

}
