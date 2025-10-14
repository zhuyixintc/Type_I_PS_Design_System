from models.features import graphs_from_smiles
from models.data_load import MPNNDataset
from models.mpnn_model import MPNNModel


# load model parameters and weights
seed = 0

task = 't1'
path = './checkpoints/' + str(task) + '/seed_' + str(seed)
model_t1 = MPNNModel()
model_t1.load_weights(path + '/model_weights.h5')

task = 'st'
path = './checkpoints/' + str(task) + '/seed_' + str(seed)
model_st = MPNNModel()
model_st.load_weights(path + '/model_weights.h5')


def predictor(smiles):
    t1, st = None, None

    print('SMILES:', smiles + ' ', end='')
    
    # transfer data
    smiles = [smiles]
    x_test = graphs_from_smiles(smiles)
    test_dataset = MPNNDataset(x_test)

    # get prediction
    for x, _ in test_dataset:
        t1 = model_t1(x).numpy()
        st = model_st(x).numpy()

    print('t1:', t1[0][0], 'st:', st[0][0])

    return t1[0][0], st[0][0]

