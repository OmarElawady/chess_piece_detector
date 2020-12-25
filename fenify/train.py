from fenify.models.cnn import CNNModel
from fenify.prepare.generate import generate_board_sample
from tqdm import tqdm
from fenify.helpers.utils import flatten
import random
random.seed(13)

train_length = 3
test_length = 3

model = CNNModel()
model.load_latest()
train_samples = [model.get_samples(generate_board_sample()) for _ in tqdm(range(train_length))]
train_samples = flatten(train_samples)
test_samples = [model.get_samples(generate_board_sample()) for _ in tqdm(range(test_length))]
test_samples = flatten(test_samples)
model.add_samples(train_samples)
model.train(200)
model.evaluate(test_samples)
model.save_snapshot()