import torch
class Classifier():
    def __init__(self, w_star):
        """
        Member Variables
        - w (torch tensor) := Parameter of size (dim , 1)
        - w_star (torch tensor) := Actual parameters of size (dim ,1)
        - c_client (float) := Control variate from the client.
        """
        self.w = None
        self.w_star = w_star
        self.c_client = None
    def obj(self, X, y, reg):
        """
        Compute the value of the loss function.
        Inputs:
        - X (torch tensor) := The data to calculate the loss on of size (N, dim).
        - y (torch tensor) := The corresponding labels of the data of size (N, 1).
        - reg (float) := The regularization parameter.
        - obj (float) := Value of the objective function.
        """

    def grad(self, X_batch, y_batch, reg):
        """
        Compute the gradient.
        Inputs:
        - X_batch (torch tensor) := The data to calculate the gradient on of size (N, dim).
        - y_batch (torch tensor) := The corresponding labels of the data of size (N, 1).
        - reg (float) := The regularization parameter.
        Outputs:
        - grad (torch tensor) := Gradient with respect to self.w of size (dim, 1).
        """

    def SGD(self, X, y, step_size=0.1, reg=1e-4, epochs=1, batch_size=50):
        """
        Vanilla SGD.
        Input:
        - X (torch tensor) := The data of size (N x dim).
        - y (torch tensor) := The labels of the data of size (N x 1).
        - step_size (float) := Step size.
        - reg (float) := Regularization parameter of objective fucntion.
        - epochs (int) := The number of passes through the data. 
        - batch_size (int) := Amount of data sampled at each iteration.
        Output:
        - obj_SGD (torch tensor) := Array of objective values at each epoch.
        - obj_SGD_iters (torch tensor) := Array of the objective values at each iteration.
        - MSE (torch tensor) := Array of the MSE at each iteration.
        """
        N, dim = X.shape
        
        obj_SGD = torch.zeros((epochs, 1))
        max_iters = int(N/batch_size)
        obj_SGD_iters = torch.zeros((int(epochs*max_iters), 1))
        MSE = torch.zeros((int(epochs*max_iters), 1))
        if self.w is None:
            self.w = 0.001 * torch.randn((dim, 1))
        for i in range(epochs):
            obj_SGD[i] += self.obj(X, y, reg).item()
            for j in range(max_iters):
                rand_idx = torch.randint(0, N-1, (batch_size, ))
                X_batch = X[rand_idx, :]
                y_batch = y[rand_idx]
                obj_SGD_iters[i*max_iters + j] += self.obj(X, y, reg).item()
                MSE[i*max_iters + j] += torch.norm(self.w - self.w_star, p=2)
                self.w = self.w - step_size * self.grad(X_batch, y_batch, reg)
        return obj_SGD, obj_SGD_iters, MSE

    def SCAFFOLD_Local(self, X, y, server_w, c_server, step_size=0.1, reg=1e-4, iterations=1, batch_size=50):
        """
        Local update for the SCAFFOLD algorithm.
        Input:
        - X (torch tensor) := The local data of size (N_local x dim).
        - y (torch tensor) := The labels of the local data of size (N_local x 1).
        - server_w (torch tensor) := The parameter of the server (dim x 1).
        - c_server (float) := Control variate of server.
        - step_size (float) := The step size of the worker.
        - reg (float) := Regularization parameter of objective function.
        - iterations (int) := The number of local iterations 
        - batch_size (int) := Amount of data sampled at each iteration.
        Output:
        - obj (torch tensor) := Array of the objective values 
        Output:
        - obj_epoch (torch tensor) := Array of the local objective values at each epoch.
        - obj_iters (torch tensor) := Array of the local objective values at each iteration.
        - MSE (torch tensor) := Array of the MSE at each iteration.
        """
        N, dim = X.shape
        # Initialize local model to take the server parameter
        self.w = server_w
        # Initialize local control variate if not initialized.
        if self.c_client is None:
            self.c_client = 0.0
        # Run local iterations.
        old_c = self.c_client
        for i in range(iterations):
            # Sample batch.
            rand_idx = torch.randint(0, N-1, (batch_size, ))
            X_batch = X[rand_idx, :]
            y_batch = y[rand_idx]
            # Do update.
            self.w = self.w - step_size * (self.grad(X_batch, y_batch, reg) - self.c_client + c_server)
        # Calculate new local control variate.
        self.c_client = self.c_client - c_server + 1 / (iterations * step_size) * (server_w - self.w)
        # Compute deviations.
        delta_w = self.w - server_w
        delta_c = self.c_client - old_c
        return delta_w, delta_c

class LogReg(Classifier):
    def __init__(self, w_star):
        super(LogReg, self).__init__(w_star)
    
    def sigmoid(self, x):
        return 1 /  (1 + torch.exp(-x))
    
    def obj(self, X, y, reg):
        N, _ = X.shape
        return 1/N * torch.sum(torch.log(1 + torch.exp(-y * X @ self.w))) + 1/2 * reg * self.w.T @ self.w 
    
    def grad(self, X_batch, y_batch, reg):
        N_batch, _ = X_batch.shape
        return 1/N_batch * X_batch.T @ (y_batch * (self.sigmoid(y_batch * X_batch @ self.w) - 1)) + reg * self.w 