objs_padding = 12

hyperparams = {
    "original-fp": #original model from pixels
        {
            "g_fc1": 256,
            "g_fc2": 256,
            "g_fc3": 256,
            "g_fc4": 256,
            
            "f_fc1": 256,
            "f_fc2": 256,
            
            "dropout": 0.5,
            "lstm_hidden": 128,
            "lstm_word_emb": 32,
            "rl_in_size": (24 + 2)*2
        },
    "original-sd": #original model state description
        {
            "g_fc1": 512,
            "g_fc2": 512,
            "g_fc3": 512,
            "g_fc4": 512,
            
            "f_fc1": 512,
            "f_fc2": 1024,
            
            "dropout": 0.05,
            "lstm_hidden": 256,
            "lstm_word_emb": 32,
            "rl_in_size": 7*2
        },
    "ir-fp": #IR model from pixels
        {
            "g_fc1": 256,
            "g_fc2": 256,
            "g_fc3": 256,
            "g_fc4": 256,
            
            "f_fc1": 256,
            "f_fc2": 256,
            
            "h_fc1": 256,
            
            "dropout": 0.5,
            "lstm_hidden": 128,
            "lstm_word_emb": 32,
            "rl_in_size": (24 + 2)*2
        },
    "ir-sd": #IR model state description
        {
            "g_fc1": 512,
            "g_fc2": 512,
            "g_fc3": 512,
            "g_fc4": 512,
            
            "f_fc1": 512,
            "f_fc2": 1024,
            
            "h_fc1": 512,
            
            "dropout": 0.05,
            "lstm_hidden": 256,
            "lstm_word_emb": 32,
            "rl_in_size": 7*2
        },
}
