from sklearn import model_selection
import pandas as pd
import numpy as np

import models
import run
import data
import utils

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = data.load_subset()
    n_users = train_x['User'].max()
    n_items = train_x['Item'].max()

    s = models.ShallowNetwork(n_items=n_items, n_users=n_users)
    history = run.fit(model=s.model,
                      train_x=train_x,
                      train_y=train_y,
                      test_x=test_x,
                      test_y=test_y)

    s.descr = s.description_str(suffix= str(max(history.epoch)+1) + "_epochs_", uid=True)

    run.plot(s.descr, history)
    #utils.save_model(s)
