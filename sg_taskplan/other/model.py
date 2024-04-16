import torch


class LeaderQNetNash(torch.nn.Module):
    """
    This class defines the leader's Q network.
    """
    def __init__(self, parameters, dims, dimAl, dimAf) -> None:
        super().__init__()
        self.dims = dims
        self.dimAl, self.dimAf = dimAl, dimAf
        
        self.n1_feature = parameters['leader']['n1_feature']
        self.n2_feature = parameters['leader']['n2_feature']

        self.linear1 = torch.nn.Linear(self.dims, self.n1_feature)
        self.linear2 = torch.nn.Linear(self.n1_feature, self.n2_feature)
        self.linear3 = torch.nn.Linear(self.n2_feature, self.dimAl*self.dimAf)
        #self.activation = torch.nn.ReLU()
        self.activation = torch.nn.LeakyReLU()

        # random initialization
        self.linear1.weight.data.normal_(mean=0., std=.1)
        self.linear1.bias.data.normal_(0., .1)
        self.linear2.weight.data.normal_(0, .1)
        self.linear2.bias.data.normal_(0, .1)
        self.linear3.weight.data.normal_(0, .1)
        self.linear3.bias.data.normal_(0, .1)

        # constant initialization
        # self.linear1.weight.data.fill_(.1)     
        # self.linear1.bias.data.fill_(.1)
        # self.linear2.weight.data.fill_(.1)
        # self.linear2.bias.data.fill_(.1)
        # self.linear3.weight.data.fill_(.1)
        # self.linear3.bias.data.fill_(.1)

    
    def forward(self, s):   
        y = self.linear1(s)
        y = self.activation(y)
        y = self.linear2(y)
        y = self.activation(y)
        y = self.linear3(y)
        return y


class FollowerQNetNash(torch.nn.Module):
    """
    This class defines the follower's Q network.
    """
    def __init__(self, parameters, dims, dimAl, dimAf) -> None:
        super().__init__()
        self.dims = dims
        self.dimAl, self.dimAf = dimAl, dimAf
        
        self.n1_feature = parameters['follower']['n1_feature']
        self.n2_feature = parameters['follower']['n2_feature']

        self.linear1 = torch.nn.Linear(self.dims, self.n1_feature)
        self.linear2 = torch.nn.Linear(self.n1_feature, self.n2_feature)
        self.linear3 = torch.nn.Linear(self.n2_feature, self.dimAl*self.dimAf)
        #self.activation = torch.nn.ReLU()
        self.activation = torch.nn.LeakyReLU()
        
        # random initialization
        self.linear1.weight.data.normal_(mean=0., std=.1)
        self.linear1.bias.data.normal_(0., .1)
        self.linear2.weight.data.normal_(0, .1)
        self.linear2.bias.data.normal_(0, .1)
        self.linear3.weight.data.normal_(0, .1)
        self.linear3.bias.data.normal_(0, .1)

        # constant initialization
        # self.linear1.weight.data.fill_(.1)     
        # self.linear1.bias.data.fill_(.1)
        # self.linear2.weight.data.fill_(.1)
        # self.linear2.bias.data.fill_(.1)
        # self.linear3.weight.data.fill_(.1)
        # self.linear3.bias.data.fill_(.1)

    
    def forward(self, s):
        y = self.linear1(s)
        y = self.activation(y)
        y = self.linear2(y)
        y = self.activation(y)
        y = self.linear3(y)
        return y


class LeaderQNetInd(torch.nn.Module):
    """
    This class defines the leader's Q network.
    """
    def __init__(self, parameters, dims, dimAl) -> None:
        super().__init__()
        self.dims = dims
        self.dimAl = dimAl
        
        self.n1_feature = parameters['leader']['n1_feature']
        self.n2_feature = parameters['leader']['n2_feature']

        self.linear1 = torch.nn.Linear(self.dims, self.n1_feature)
        self.linear2 = torch.nn.Linear(self.n1_feature, self.n2_feature)
        self.linear3 = torch.nn.Linear(self.n2_feature, self.dimAl)
        #self.activation = torch.nn.ReLU()
        self.activation = torch.nn.LeakyReLU()

        # random initialization
        self.linear1.weight.data.normal_(mean=0., std=.1)
        self.linear1.bias.data.normal_(0., .1)
        self.linear2.weight.data.normal_(0, .1)
        self.linear2.bias.data.normal_(0, .1)
        self.linear3.weight.data.normal_(0, .1)
        self.linear3.bias.data.normal_(0, .1)

        # constant initialization
        # self.linear1.weight.data.fill_(.1)     
        # self.linear1.bias.data.fill_(.1)
        # self.linear2.weight.data.fill_(.1)
        # self.linear2.bias.data.fill_(.1)
        # self.linear3.weight.data.fill_(.1)
        # self.linear3.bias.data.fill_(.1)

    
    def forward(self, s):   
        y = self.linear1(s)
        #y = self.bn1(y)
        y = self.activation(y)
        y = self.linear2(y)
        #y = self.bn2(y)
        y = self.activation(y)
        y = self.linear3(y)
        return y


class FollowerQNetInd(torch.nn.Module):
    """
    This class defines the follower's Q network.
    """
    def __init__(self, parameters, dims, dimAf) -> None:
        super().__init__()
        self.dims = dims
        self.dimAf = dimAf

        self.n1_feature = parameters['follower']['n1_feature']
        self.n2_feature = parameters['follower']['n2_feature']

        self.linear1 = torch.nn.Linear(self.dims, self.n1_feature)
        self.linear2 = torch.nn.Linear(self.n1_feature, self.n2_feature)
        self.linear3 = torch.nn.Linear(self.n2_feature, self.dimAf)
        #self.activation = torch.nn.ReLU()
        self.activation = torch.nn.LeakyReLU()

        # random initialization
        self.linear1.weight.data.normal_(mean=0., std=.1)
        self.linear1.bias.data.normal_(0., .1)
        self.linear2.weight.data.normal_(0, .1)
        self.linear2.bias.data.normal_(0, .1)
        self.linear3.weight.data.normal_(0, .1)
        self.linear3.bias.data.normal_(0, .1)

        # constant initialization
        # self.linear1.weight.data.fill_(.1)     
        # self.linear1.bias.data.fill_(.1)
        # self.linear2.weight.data.fill_(.1)
        # self.linear2.bias.data.fill_(.1)
        # self.linear3.weight.data.fill_(.1)
        # self.linear3.bias.data.fill_(.1)

    
    def forward(self, s):
        y = self.linear1(s)
        y = self.activation(y)
        y = self.linear2(y)
        y = self.activation(y)
        y = self.linear3(y)
        return y

