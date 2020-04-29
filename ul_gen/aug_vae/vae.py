import torch

class Reshape(torch.nn.Module):
  def __init__(self, output_shape):
    super(Reshape, self).__init__()
    self.output_shape = output_shape

  def forward(self, x):
    return x.view(*((len(x),) + self.output_shape))

class PrintNode(torch.nn.Module):
  def __init__(self, identifier="print:"):
    super(PrintNode, self).__init__()
    self.identifier = identifier

  def forward(self, x):
    print(self.identifier, x.shape)
    return x

class VAE(torch.nn.Module):

    def __init__(self, img_dim=64, img_channels=3, z_dim=32, final_act="tanh"):
        super(VAE, self).__init__()
        self.img_dim = img_dim
        self.z_dim = z_dim
        self.img_channels = img_channels

        final_act_fn = lambda: torch.nn.Tanh() if final_act == "tanh" else torch.nn.Sigmoid()

        final_feature_dim = img_dim // (2**4)
        self.encoder_net = torch.nn.Sequential(torch.nn.Conv2d(self.img_channels, 32, kernel_size=4, stride=2, padding=1),
                                            # PrintNode("Enc Conv1"),
                                            torch.nn.ReLU(),
                                            torch.nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
                                            # PrintNode("Enc Conv2"),
                                            torch.nn.ReLU(),
                                            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                                            # PrintNode("Enc Conv3"),
                                            torch.nn.ReLU(),
                                            torch.nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
                                            # PrintNode("Enc Conv4"),
                                            torch.nn.ReLU(),
                                            torch.nn.Flatten(),
                                            # PrintNode("Flattened Enc"),
                                            torch.nn.Linear(final_feature_dim*final_feature_dim*64, 2*self.z_dim),
                                            # PrintNode("Enc Z Out"),
                                            )
        
        self.decoder_net = torch.nn.Sequential(torch.nn.Linear(self.z_dim, 256),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(256, final_feature_dim*final_feature_dim*64),
                                            Reshape((64, final_feature_dim, final_feature_dim)),
                                            # PrintNode("Dec Conv In"),
                                            torch.nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
                                            # PrintNode("Dec Conv1"),
                                            torch.nn.ReLU(),
                                            torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                                            # PrintNode("Dec Conv2"),
                                            torch.nn.ReLU(),
                                            torch.nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
                                            # PrintNode("Dec Conv3"),
                                            torch.nn.ReLU(),
                                            torch.nn.ConvTranspose2d(32, self.img_channels, kernel_size=4, stride=2, padding=1),
                                            # PrintNode("Dec Conv4"),
                                            final_act_fn(),
                                            )

    def encoder(self, x):
        mu, log_var = torch.chunk(self.encoder_net(x), 2, dim=1)
        return mu, log_var
        
    def decoder(self, z):
        return self.decoder_net(z)
        
    def forward(self, x, deterministic=False):
        mu, log_var = self.encoder(x)
        if deterministic:
            z = mu
        else:
            z = torch.exp(0.5*log_var) * torch.randn_like(mu) + mu
        x_hat = self.decoder(z)
        return x_hat, mu, log_var