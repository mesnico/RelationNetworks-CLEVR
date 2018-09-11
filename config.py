import numpy as np
from ray.tune.variant_generator import grid_search

hyperparams = {
    "sd-innetaggr": {
        "repeat": 1,
        "resources": { "cpu": 4, "gpu": 1},
        "tune": {
            "state_description": True,
            "g_layers": [
                512,
                512,
                512,
                grid_search([512, 1024, 2048]),
                grid_search([256, 512]),
                512,
                1024
            ],
            "aggregation_position": 4, #lambda spec: np.random.randint(1, 4),
            "question_injection_position": grid_search([4, 5]), #lambda spec: spec.config.aggregation_position + np.random.randint(1, 3),
            "dropouts": {
                "4": grid_search([0.1, 0.2, 0.3, 0.4]),
            },
                
            "lstm_hidden": 256,
            "lstm_word_emb": 32,
            "rl_in_size": 14
        }
    }

}
