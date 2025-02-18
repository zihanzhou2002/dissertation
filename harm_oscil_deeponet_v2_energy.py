from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch import nn


import harm_oscil_system
import deepxde as dde
from spaces import FinitePowerSeries, FiniteChebyshev, GRF
from system import LTSystem, ODESystem, DRSystem, CVCSystem, ADVDSystem
from utils import merge_values, trim_to_65535, mean_squared_error_outlier, safe_test


def test_u_ode(nn, system, T, m, model, data, u, fname, num=100):
    """Test ODE"""
    sensors = np.linspace(0, T, num=m)[:, None]
    sensor_values = u(sensors)
    x = np.linspace(0, T, num=num)[:, None]
    #X_test = [np.tile(sensor_values.T, (num, 1)), x]
    X_test = (np.tile(sensor_values.T, (num, 1)), x)
    
    print("Testing the ODE")
    
    y_test = system.eval_s_func(u, x)
    
    #print(f"X_test \n{X_test}")
    #print(f"y_test \n{y_test}")
    if nn != "opnn":
        X_test = merge_values(X_test)
    
    #y_pred = model.predict(data.transform_inputs(X_test))
    y_pred = model.predict(X_test)
    
    
    #print("Shape of X_test", len(X_test))
    #print("Shape of y_test", len(y_test))
    
    np.savetxt(fname, np.hstack((x, y_test, y_pred)))
    
    #print("Shape of y_pred", len(y_pred))
    
    np.savetxt(fname, np.hstack((x, y_test, y_pred)))
    print("L2relative error:", dde.metrics.l2_relative_error(y_test, y_pred))
    print("MSE error:", dde.metrics.mean_squared_error(y_test, y_pred))

def plot_energies(model, X_test,y_test, omega):
    def is_nonempty(X):
        return len(X[0]) > 0 if isinstance(X, (list, tuple)) else len(X) > 0
    
    time = X_test[1]
    y_pred = model(X_test).detach().numpy()
    energies_pred = 0.5*(y_pred[:,1]**2 + (omega*y_pred[:, 0])**2)
    energies_true = 0.5*(y_test[:,1]**2 + (omega*y_test[:, 0])**2)
    
    plt.plot(time, energies_pred, label='predicted energies')
    plt.plot(time, energies_true, label='actual energies')
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.ylim(-5,5)
    plt.legend()
    plt.title("Total Energies over Time")
    plt.savefig("deeponet/plots/total_energies.png")
    plt.show()

def plot_prediction(model, X_test,y_test, omega):
    def is_nonempty(X):
        return len(X[0]) > 0 if isinstance(X, (list, tuple)) else len(X) > 0
    
    
    X = X_test[0]
    time = X_test[1]
    y_pred = np.zeros(len(time))

    harm_pred = y_pred[:,0]
    plt.plot(harm_pred, marker='o')
    plt.xlabel("Time")
    plt.ylabel("x")
    plt.title("Predicted_harmonic_oscillator")
    plt.savefig(f"deeponet/plots/pred_{omega}.png")
    plt.show()


    

def main():
    def model(X): # take in num_datax1
        x_func=net.branch(torch.tensor(X[0])) # output num_datax 2x2, num_datax 4
        # num_datax4
        #b=torch.reshape(b, (2, 2)) #num_datax 2x2
        #print(f"x_fun = {x_func}")
        x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1])))
        #print(f"x_loc = {x_loc}")
        
        # Split x_func into respective outputs
        shift = 0
        size = x_loc.shape[1]
        xs = []
        for i in range(net.num_outputs):
            x_func_ = x_func[:, shift : shift + size]
            x = net.merge_branch_trunk(x_func_, x_loc, i)
            xs.append(x)
            shift += size
        
        result = net.concatenate_outputs(xs)
        #print(f"result = {result}")
        return result

    def model_energy_v2(X): # take in num_datax1
        x_func=net.branch(torch.tensor(X[0])) # output num_datax 2x2, num_datax 4
        
        x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1])))
        W = torch.tensor([[omega**2, 0], [0, 1]]).double()
        W_sqrt = torch.tensor([[omega, 0], [0, 1]]).double()
        # Split x_func into respective outputs

        xs = []
        for i in range(len(x_func)):
            p0, q0 = X[0][i]
            x_func_ = torch.reshape(x_func[i],(2, -1) ).double()
            x_loc_ = x_loc[i].double()
            
            E = 0.5* (omega*p0**2 +q0**2)
            B_tilde = torch.mm(W_sqrt, x_func_).double()
            Q_tilde, R = torch.linalg.qr(B_tilde)
            alpha_tilde = torch.mm(R, x_loc_.unsqueeze(1)).double()
            
            Q_tilde = Q_tilde.to(torch.float64)
            R = R.to(torch.float64)
            alpha_scaled = alpha_tilde* np.sqrt(E) / torch.linalg.vector_norm(alpha_tilde)
            W_sqrt_inv = torch.linalg.inv(W_sqrt)
            
            W_sqrt_inv = W_sqrt_inv.to(torch.float64)
            temp = torch.mm(W_sqrt_inv, Q_tilde).double()
            x = torch.mm(temp, alpha_scaled).double() + torch.tensor([net.b[0], net.b[1]], dtype = torch.float64).unsqueeze(1)
            xs.append(x.squeeze())

        result = torch.stack(xs, dim=0).to(torch.float64)

        return result

    def model_energy_v2_batch(X): # take in num_datax1
        x_func=net.branch(torch.tensor(X[0])) # output num_datax 2x2, num_datax 4
        
        x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1])))
        # Split x_func into respective outputs
        
        x_func = x_func.view(x_func.shape[0], 2, -1).to(torch.float64)
        x_loc = x_loc.unsqueeze(2).to(torch.float64)

        p0 = torch.tensor(X[0][:, 0])
        q0 = torch.tensor(X[0][:, 1])
        
        E = 0.5* ((omega*p0)**2 +q0**2)
        
        W_sqrt = torch.tensor([[omega, 0], [0, 1]]).double()
        W_sqrt_inv = torch.linalg.inv(W_sqrt)
        W_sqrt_batch = W_sqrt.expand(x_func.shape[0], -1, -1)
        W_sqrt_inv_batch = W_sqrt_inv.expand(x_func.shape[0], -1, -1)
        
        B_tilde = torch.bmm(W_sqrt_batch, x_func).double()
        
        Q_tilde, R = torch.linalg.qr(B_tilde)
        
        alpha_tilde = torch.bmm(R, x_loc).double()
        
        b = torch.tensor([net.b[0], net.b[1]], dtype = torch.float64).unsqueeze(1)
        b_batch = b.expand(x_func.shape[0], -1, -1)
        
        norm_alpha_tilde = torch.linalg.vector_norm(alpha_tilde, dim=1, keepdim=True)
        alpha_scaled = alpha_tilde* torch.sqrt(2*E).unsqueeze(1).unsqueeze(2) / norm_alpha_tilde
        
        basis = torch.bmm(W_sqrt_inv_batch, Q_tilde).double()
        result = torch.bmm(basis, alpha_scaled) + b_batch
        
        return result.squeeze()
    
    def model_energy(X): # take in num_datax1
        x_func=net.branch(torch.tensor(X[0])) # output num_datax 2x2, num_datax 4
        
        x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1])))
        # Split x_func into respective outputs

        xs = []
        for i in range(len(x_func)):
            p0, q0 = X[0][i]
            x_func_ = torch.reshape(x_func[i],(2, -1) ).double()
            x_loc_ = x_loc[i].double()
            
            E = 0.5* (omega*p0**2 +q0**2)
            
            Q, R = torch.linalg.qr(x_func_)
            alpha_tilde = torch.mm(R, x_loc_.unsqueeze(1)).double()
            
            Q = Q.to(torch.float64)
            R = R.to(torch.float64)
            alpha_scaled = alpha_tilde* np.sqrt(E) / torch.linalg.vector_norm(alpha_tilde)
            
            b = torch.tensor([net.b[0], net.b[1]], dtype = torch.float64).unsqueeze(1)

            x = torch.mm(Q, alpha_scaled).double() + b
            xs.append(x.squeeze())

        result = torch.stack(xs, dim=0).to(torch.float64)
        return result

    def model_energy_batch(X): # take in num_datax1
        x_func=net.branch(torch.tensor(X[0])) # output num_datax 2x2, num_datax 4
        
        x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1])))
        # Split x_func into respective outputs
        
        x_func = x_func.view(x_func.shape[0], 2, -1).to(torch.float64)
        x_loc = x_loc.unsqueeze(2).to(torch.float64)

        p0 = torch.tensor(X[0][:, 0])
        q0 = torch.tensor(X[0][:, 1])
        
        E = 0.5* (omega*p0**2 +q0**2)

        Q, R = torch.linalg.qr(x_func)
        
        alpha_tilde = torch.bmm(R, x_loc).double()
        
        b = torch.tensor([net.b[0], net.b[1]], dtype = torch.float64).unsqueeze(1)
        b_batch = b.expand(x_func.shape[0], -1, -1)
        
        norm_alpha_tilde = torch.linalg.vector_norm(alpha_tilde, dim=1, keepdim=True)
        alpha_scaled = alpha_tilde* torch.sqrt(2*E).unsqueeze(1).unsqueeze(2) / norm_alpha_tilde
        
        result_energy = torch.bmm(Q, alpha_scaled) + b_batch
        result_energy = result_energy.double()
        result_orig = torch.bmm(x_func, x_loc) + b_batch
        result_orig = result_orig.double()
        return result_energy.squeeze()

    def model_energy_v3(X): # take in num_datax1
        x_func=net.branch(torch.tensor(X[0])) # output num_datax 2x2, num_datax 4
        
        x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1])))
        # Split x_func into respective outputs

        xs = []
        for i in range(len(x_func)):
            p0, q0 = X[0][i]
            x_func_ = torch.reshape(x_func[i],(2, -1) ).double()
            x_loc_ = x_loc[i].double()
            
            E = 0.5* (omega*p0**2 +q0**2)
            
            Q, R = torch.linalg.qr(x_func_)
            alpha1 = torch.mm(R, x_loc_.unsqueeze(1)).double()
            
            Q = Q.to(torch.float64)
            R = R.to(torch.float64)
            
            b = torch.tensor([net.b[0], net.b[1]], dtype = torch.float64).unsqueeze(1)
            alpha2 = torch.linalg.solve(Q, b)
            alpha2 = alpha2.to(torch.float64)
            
            alpha_tilde = alpha1 + alpha2
            
            alpha_scaled = alpha_tilde* np.sqrt(E) / torch.linalg.vector_norm(alpha_tilde)
            x = torch.mm(Q, alpha_scaled).double() 
            xs.append(x.squeeze())

        #print(f"xs {len(xs), len(xs[0])}")
        #result = net.concatenate_outputs(xs)
        #print(f"result = {result}")
        result = torch.stack(xs, dim=0).to(torch.float64)
        #print(f"result: {result.shape}")
        return result

    def model_energy_v3_batch(X): # take in num_datax1
        x_func=net.branch(torch.tensor(X[0])) # output num_datax 2x2, num_datax 4
        
        x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1])))
        # Split x_func into respective outputs
        
        x_func = x_func.view(x_func.shape[0], 2, -1).to(torch.float64)
        x_loc = x_loc.unsqueeze(2).to(torch.float64)

        p0 = torch.tensor(X[0][:, 0])
        q0 = torch.tensor(X[0][:, 1])
        
        E = 0.5* (omega*p0**2 +q0**2)

        Q, R = torch.linalg.qr(x_func)
        
        alpha1 = torch.bmm(R, x_loc).double()
        
        b = torch.tensor([net.b[0], net.b[1]], dtype = torch.float64).unsqueeze(1)
        b_batch = b.expand(x_func.shape[0], -1, -1)
        
        alpha2 = torch.linalg.solve(Q, b_batch).double()
        
        alpha_tilde = alpha1 + alpha2
        norm_alpha_tilde = torch.linalg.vector_norm(alpha_tilde, dim=1, keepdim=True)
        alpha_scaled = alpha_tilde* torch.sqrt(E).unsqueeze(1).unsqueeze(2) / norm_alpha_tilde
        
        result = torch.bmm(Q, alpha_scaled).double()
        return result.squeeze(2)
        

    def model_matrix(X): # take in num_datax1
        x_func=net.branch(torch.tensor(X[0])) # output num_datax 2x2, num_datax 4
        
        x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1])))
        W = torch.tensor([[omega, 0], [0, 1]])
        W_sqrt = torch.tensor([[np.sqrt(omega), 0], [0, 1]])
        # Split x_func into respective outputs

        xs = []
        for i in range(len(x_func)):
            x_func_ = torch.reshape(x_func[i],(2, -1) )
            x_loc_ = x_loc[i]
            x = torch.mm(x_func_, x_loc_.unsqueeze(1)) + torch.tensor([net.b[0], net.b[1]]).unsqueeze(1)
            xs.append(x.squeeze())

        #print(f"xs {len(xs), len(xs[0])}")
        #result = net.concatenate_outputs(xs)
        #print(f"result = {result}")
        result = torch.stack(xs, dim=0)
        #print(f"result: {result.shape}")
        return result

    def model_matrix_batch(X): # take in num_datax1
        x_func=net.branch(torch.tensor(X[0])) # output num_datax 2x2, num_datax 4
        
        x_loc = net.activation_trunk(net.trunk(torch.tensor(X[1])))

        x_func = x_func.view(x_func.shape[0], 2, -1).to(torch.float64)
        x_loc = x_loc.unsqueeze(2).to(torch.float64)
        
        b =  torch.tensor([net.b[0], net.b[1]], dtype=torch.float64).unsqueeze(1)
        b_batch = b.expand(x_func.shape[0], -1, -1).to(torch.float64)
        
        result = torch.bmm(x_func, x_loc).to(torch.float64) + b_batch
        #print(f"result: {result.shape}")
        return result.squeeze()

    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    problem = "ode"
    
    # harm oscil parameters
    m = 2
    T = 5
    omega = 2
    L = 2
    
    # for testing
    num_train = 300
    num_test = 100
    lr = 0.001
    epochs = 50000
    #epochs = 1
    #epochs = 500

    # Network
    activation = "relu"
    initializer = "Glorot normal"  # "He normal" or "Glorot normal"
    dim_x = 1 if problem in ["ode", "lt"] else 2
    
   
    net = dde.nn.DeepONet(
        [m, 40, 40],
        [dim_x, 40, 20],
        activation,
        initializer,
        num_outputs = 2,
        #multi_output_strategy="independent_energy"# For harmonic oscillator
        #ulti_output_strategy="split_both_energy"
        multi_output_strategy="split_branch"
    )       
        
    

    
    # Generate dataset
    #X_train, y_train = harm_oscil_system.gen_harm_dataset(1,1, omega, T, num_train) 
    X_train, y_train = harm_oscil_system.gen_harm_dataset(omega, L, T, num_train) 
    #X_train, y_train = harm_oscil_system.gen_harm_dataset_fixed(1.5,1, omega, T, num_train) 
    X_test, y_test = harm_oscil_system.gen_harm_dataset_fixed(1.5,1, omega,T, num_test)
    print(X_train,y_train)
    print("Dataset generated")
    # Create dataloaders
    output = model_energy_v2_batch(X_train)
    batch_size = 4
    
    
    
    #y_pred = model_energy_v3_batch(X_train)
    
    #y_pred = model(X_test)
    
    #y_pred = model_matrix_batch(X_train)
    # define optimizer 
    # Custom backprop
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    print("Initialized ")
    print("Start training")
    
    p0 = torch.tensor(X_train[0][:, 0])
    q0 = torch.tensor(X_train[0][:, 1])
        
    E = 0.5* (omega*p0**2 +q0**2)
    # Training the model
  
    for epoch in range(epochs):
        optimizer.zero_grad()
        #y_pred = model(X_train)
        y_pred_energy = model_energy_v2_batch(X_train)
        
        loss = loss_fn(y_pred_energy, torch.tensor(y_train, dtype=torch.float64))
        loss.backward()
        optimizer.step()
        
        
        
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1}, loss {loss.item() :.8f} ")
            #print(y_pred_energy)
            #print(f"Without energy - \n Energy fluctuation: {torch.linalg.vector_norm(0.5*(omega *y_pred_orig[:,0]**2 + y_pred_orig[:,1]**2) - E)}")
            #print(f"With energy - \n Energy fluctuation: {torch.linalg.vector_norm(0.5*(omega *y_pred_energy[:,0]**2 + y_pred_energy[:,1]**2) - E)}")
            #print(f"Differnce = {y_pred_energy - y_pred_orig}")

        
    
    print("Finished Training")
    
    # Testing
    y_pred = model_energy_v2_batch(X_test)
    #print(f"y_pred = {y_pred}")
    plt.plot(X_test[1], y_pred[:,0].detach().numpy(), label = 'prediction')
    plt.plot(X_test[1], y_test[:,0], label = 'ground truth')
    plt.legend()
    plt.savefig("deeponet/plots/harm_prediction.png")
    plt.show()
    
    #safe_test(model, data, X_test, y_test)
    
    #plot_energies(model, X_test, omega)
    #plot_prediction(model, X_test, omega)
    plot_energies(model_energy_batch, X_test,y_test, omega)

if __name__ == "__main__":
    main()
