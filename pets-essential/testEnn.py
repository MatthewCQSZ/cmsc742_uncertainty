
from enn import losses
from enn import networks
from enn import supervised
from enn.supervised import regression_data
import optax

# A small dummy dataset
dataset = regression_data.make_dataset()

print(type(dataset))
data = dataset.__next__()
x = data.x
print(x.shape)
y = data.y
print(y.shape)

# ENN
enn = networks.MLPEnsembleMatchedPrior(
    output_sizes=[50, 50, 1],
    num_ensemble=10,
    dummy_input = x[0]
)

# Loss
loss_fn = losses.average_single_index_loss(
    single_loss=losses.L2Loss(),
    num_index_samples=10
)

print("Done")

# Optimizer
optimizer = optax.adam(1e-3)

# Train the experiment
experiment = supervised.Experiment(
    enn, loss_fn, optimizer, dataset, seed=0, logger=None)
experiment.train(1000)


#print(data.shape)