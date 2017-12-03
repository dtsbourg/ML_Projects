from keras.utils import plot_model

def save_model_graph(model):
    plot_model(model, to_file='../res/model/' + model.descr + '.png')

def save_model(cm):
    try:
        cm.model.save('../res/model/'+cm.descr+'.h5')
    except Exception as e:
        raise
