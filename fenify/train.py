from fenify.models.basic import MnistModel
from fenify.prepare.generate import generate_board_sample
from tqdm import tqdm
from fenify.helpers.utils import flatten

train_length = 6
test_length = 2

model = MnistModel()
model.load_latest()
train_samples = [model.get_samples(generate_board_sample()) for _ in tqdm(range(train_length))]
train_samples = flatten(train_samples)
test_samples = [model.get_samples(generate_board_sample()) for _ in tqdm(range(test_length))]
test_samples = flatten(test_samples)
model.add_samples(test_samples)
model.train(1000)
model.evaluate(test_samples)
model.save_snapshot()