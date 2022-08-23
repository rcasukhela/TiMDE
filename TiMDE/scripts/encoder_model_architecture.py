import torch

class AE(torch.nn.Module):
    '''
    encoder model.
    '''
    def __init__(self):
        super().__init__()
         
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 512*768 ==> 1
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(256*256 + 1, 200*200),
            torch.nn.ReLU(),
            torch.nn.Linear(200*200, 150*150),
            torch.nn.ReLU(),
            torch.nn.Linear(150*150, 128*128),
            torch.nn.ReLU(),
            torch.nn.Linear(128*128, 100*100),
            torch.nn.ReLU(),
            torch.nn.Linear(100*100, 75*75),
            torch.nn.ReLU(),
            torch.nn.Linear(75*75, 50*50),
            torch.nn.ReLU(),
            torch.nn.Linear(50*50, 25*25),
            torch.nn.ReLU(),
            torch.nn.Linear(25*25, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
            torch.nn.ReLU(),
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        return encoded