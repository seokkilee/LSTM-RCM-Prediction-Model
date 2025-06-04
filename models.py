import torch
import gpytorch

class LSTMNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True, 
            bidirectional=False, dropout=0.2
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 512), torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512), torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 256), torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, output_dim)
        )
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1]
        return self.fc(x)

class SVGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, num_tasks, output_dim):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0), batch_shape=torch.Size([num_tasks])
        )
        variational_strategy = gpytorch.variational.MultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=output_dim, batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks])
        )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
