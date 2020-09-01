import unidecode
import string
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils import data
from torch import nn, optim

DATASET_PATH = 'data/arXiv/'
BATCH_SIZE = 32

all_characters = string.printable
char_to_index_dict = {c: i for i, c in enumerate(all_characters)}


class PaperDataset(data.Dataset):
    """
    Data loader for the arXiv Title and Abstracts Dataset. 
    """

    def __init__(self, device, split="train", chunk_len=200, data_dir=DATASET_PATH):
        assert (split in ["train", "val", "test"])
        file_path = os.path.join(data_dir, split + '.txt')
        self.text = unidecode.unidecode(open(file_path).read())
        self.classes = all_characters
        UNK = string.printable.index(' ')
        self.text_ind = map(lambda x: char_to_index_dict.get(x, UNK), self.text)
        self.text_ind = np.array(list(self.text_ind))
        self.text_tensor = torch.from_numpy(self.text_ind).long().to(device)
        self.chunk_len = chunk_len

    def set_offset(self, offset=0, seed=None):
        if offset is not None:
            self.offset = offset
        else:
            assert (seed is not None)
            rng = np.random.RandomState(seed)
            self.offset = rng.randint(self.chunk_len)

    def __len__(self):
        num_valid_starts = len(self.text_tensor) - self.offset - self.chunk_len - 1
        num_chunks = num_valid_starts // self.chunk_len
        return num_chunks

    def __getitem__(self, index):
        index = index * self.chunk_len + self.offset
        inp = self.text_tensor[index:index + self.chunk_len]
        target = self.text_tensor[index + 1:index + self.chunk_len + 1]
        return inp, target


# We are providing a very simple network that looks at just the current token,
# and tries to predict the next token.  As part of the assignment you will
# develop a more expressive model to improve performance.
class OneGram(nn.Module):
    def __init__(self, n_classes):
        super(OneGram, self).__init__()
        self.layers = nn.Sequential(
            nn.Embedding(n_classes, n_classes)
        )

    def init_hidden(self, batch_size, device):
        hidden = []
        return hidden

    def forward(self, inp, hidden):
        x = self.layers(inp)
        return x, hidden


def generate_abstract(model, device, file_name, temperature=0.8):
    # get list of titles from test set
    text = unidecode.unidecode(open(os.path.join(DATASET_PATH, "test.txt")).read())
    titles = [line for line in text.split("\n") if line[:6] == "Title:"]
    
    # get random indices for pulling titles from test set
    rng = np.random.RandomState(111)
    indices = [i for i in range(len(titles))]
    rng.shuffle(indices)
    
    # title prompts for generating abstracts
    prompts = [titles[indices[i]] + "\nAbstract: " for i in range(16)]
    abstracts = [evaluate(model, device, prime_str=prompt, predict_len=None, temperature=temperature)
                 for prompt in prompts]

    with open(file_name, "w") as f:
        for i, abstract in enumerate(abstracts):
            f.write(abstract + "\n")


def evaluate(model, device, prime_str='A', predict_len=200, temperature=0.8):
    """
    Runs the model to generate text. First passes in the prime_str to
    initialize the hidden state of the model, and then generates one character
    at a time by sampling from the probabilities predicted by the model.
    Generated characters are fed back into the model as the context for the
    model to condition on.
    prime_str:   string to warm up the model with
    predict_len: how long a text to generate. If None, generates till 4096
                 characters or when a \n is produced.
    temperature: temperature for the softmax for sampling, higher temperature
                 will lead to more diverse outputs
    """
    model.eval()
    UNK = string.printable.index(' ')
    stop_at_newline = False
    prime_ind = np.array(list(map(lambda x: char_to_index_dict.get(x, UNK), prime_str)))
    prime_tensor = torch.from_numpy(prime_ind).long().to(device)
    predicted_str = prime_str

    # My simple 1-Gram model doesn't require any hidden state, but your RNN
    # model will require initialization of hidden state.
    hidden = model.init_hidden(1, device)

    # My simple 1-Gram model, only looks at the last one letter, but your more
    # expressive RNN model could look at the entire context. Something as
    # follows:
    for p in range(len(prime_str) - 1):
        _, hidden = model(prime_tensor[p].unsqueeze(0).unsqueeze(0), hidden)

    inp = prime_tensor[-1].unsqueeze(0).unsqueeze(0)

    if predict_len is None:
        predict_len = 4096
        stop_at_newline = True

    for p in range(predict_len):
        output, hidden = model(inp, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted_str += predicted_char
        if stop_at_newline and predicted_char == "\n":
            break
        inp = top_i.unsqueeze(0).unsqueeze(0)
    return predicted_str


# We have provided a basic training loop below with val and train functions.
# Feel free to modify the training loop, to suit your needs.
def simple_val(model, imset, device):
    val_dataset = PaperDataset(device, split=imset)
    val_dataset.set_offset(offset=0, seed=None)
    val_dataloader = data.DataLoader(val_dataset, batch_size=1,
                                     shuffle=False, num_workers=0,
                                     drop_last=False)
    preds, gts = [], []

    # Put model in evaluation mode.
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='mean')
    total_loss = []
    for i, batch in enumerate(val_dataloader):
        hidden = model.init_hidden(1, device)
        inp, gt = batch
        pred, hidden = model(inp, hidden)
        pred = pred.permute(0, 2, 1)

        loss = criterion(pred, gt)
        pred = torch.softmax(pred, 1)
        total_loss.append(loss.detach().cpu().numpy())
    print('Val Loss: {:5.4f}'.format(np.mean(total_loss)))

    # Put model back in training mode
    model.train()


def simple_train(device):
    # Initializing a simple model.
    train_dataset = PaperDataset(device, split='train')
    model = OneGram(len(train_dataset.classes))
    # model = NGram(len(train_dataset.classes), 8)

    # Moving the model to the device (eg GPU) used for training.
    model = model.to(device)

    # Setting the model in training mode.
    model.train()

    # Loss criterion that will be used to train the model
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # Optimizer that we will be using, along with the learning rate. Feel free
    # to experiment with a different learning rate, optimizer.
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    iteration = 0

    # Number of epochs to train for. You mat need to train for much longer.
    num_epochs = 200
    for j in tqdm(range(num_epochs)):
        train_dataset.set_offset(offset=None, seed=j)
        train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                           shuffle=True, num_workers=0,
                                           drop_last=True)
        training_loss = []
        for i, batch in enumerate(train_dataloader):
            # Zero out gradient blobs in the optimizer
            optimizer.zero_grad()

            hidden = model.init_hidden(BATCH_SIZE, device)
            inp, gt = batch

            # Get model predictions
            pred, hidden = model(inp, hidden)

            # Compute loss against gt
            pred = pred.permute(0, 2, 1)
            loss = criterion(pred, gt)

            # Compute gradients with respect to loss
            loss.backward()

            # Take a step to update network parameters.
            optimizer.step()
            iteration += 1

            l = loss.detach().cpu().numpy()
            training_loss += [l]

        # Every few epochs print the loss.
        if np.mod(j + 1, 20) == 0:
            print('Loss [{:8d}]: {:5.4f}'.format(iteration, np.mean(training_loss)))

        # Every few epochs get metrics on the validation set.
        if np.mod(j + 1, 40) == 0:
            simple_val(model, 'val', device)
            # Generate some text
            gen_text = evaluate(model, device, prime_str='Title: ',
                                predict_len=512, temperature=0.8)
            print('Sample Generated Text:')
            print(gen_text)
            model.train()

    # We are simply returning the model after 20 epocs, you may want to train
    # for longer, and possibly pick the model that leads to the best metrics on
    # the validation set.
    return model


def main():
    device = torch.device('cuda:0')
    model = simple_train(device)

    # Next, we evaluate the model. Here we are evaluating on the validation
    # set. Evaluation code produces NLL and some sample text.  During
    # development, you should evaluate on the validation set, and identify
    # which variant works the best.  Document your observations (variant tried,
    # and performance obtained on the validation set) in your PDF report. Once
    # you are happy with the performance of your model, you should test on the
    # test set, and report the NLL that you obtain on the test set.
    simple_val(model, 'val', device)

    # Following function generates abstracts for titles in the test set for
    # qualitative analysis.
    file_name = 'OneGram-generated-abstracts.txt'
    generate_abstract(model, device, file_name, temperature=0.8)


if __name__ == '__main__':
    main()
