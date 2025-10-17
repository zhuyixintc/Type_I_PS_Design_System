from models.features import graphs_from_smiles
from models.data_load import MPNNDataset
from models.mpnn_model import MPNNModel


# load model parameters and weights
seed = 0

# p1 and p2 are two properties
task = 'p1'
path = 'your saved path for p1'
model_t1 = MPNNModel()
model_t1.load_weights()

task = 'p2'
path = 'your saved path for p2'
model_st = MPNNModel()
model_st.load_weights()


def predictor(smiles):
    p1, p2 = None, None

    print('SMILES:', smiles + ' ', end='')
    
    # transfer data
    smiles = [smiles]
    x_test = graphs_from_smiles(smiles)
    test_dataset = MPNNDataset(x_test)

    # get prediction
    for x, _ in test_dataset:
        p1 = model_t1(x).numpy()
        p2 = model_st(x).numpy()

    print('p1:', p1[0][0], 'st:', p2[0][0])

    return p1[0][0], p2[0][0]

