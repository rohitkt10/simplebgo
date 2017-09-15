import numpy as np
import os
from pdb import set_trace as keyboard
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_context('paper')
sns.set_context(rc={'lines.markeredgewidth': 1.5})
import GPy
from design import latin_center
from GPy.kern import RBF
from GPy.models import GPRegression
from scipy.stats import norm
from scipy.interpolate import griddata

class BayesianOptimizer(object):
    def __init__(self, objfunc, bounds, kernel, 
        Xinit=None, Yinit=None, 
        num_init = 10, 
        maxiter = 15, 
        minimize = True, 
        ei_opt_points = 10000, 
        num_restarts = 10,
        verbose=True, 
        plot=True,
        resultdir=os.getcwd()):
        """
        :param objfunc: The objective function to optimize. 
        :param bounds: The bounds of  the  input variables as a list 
                        of tuples. 
        :param verbose: Whether to print out status of optimization after 
                    every iteration. (default: True)
        :param minimize: Whether to minimize the objective function (default: True)
        :param plot: Whether to plot the EI and GP surrogate plots (default: True)

        """
        if Xinit is not None: 
            assert Xinit.ndim == 2
        if Yinit is not None:
            assert Yinit.ndim == 2
        if minimize:
            self.objfunc_arg = objfunc
            self.objfunc = lambda x:-self.objfunc_arg(x)
        else:
            self.objfunc = objfunc
        self.bounds = bounds 
        self.verbose = verbose
        self.dim = len(bounds)
        self.ei_opt_points = ei_opt_points
        self.Xinit = Xinit
        self.Yinit = Yinit
        self.kernel = kernel
        self.num_restarts = num_restarts
        self.maxiter = maxiter
        self.verbose = verbose
        self.plot = plot
        self.resultdir = resultdir
        self.current_file_path =  os.path.join(os.path.dirname(os.path.realpath(__file__)), 'bgo.py')
        if not os.path.exists(self.resultdir):
            os.makedirs(self.resultdir)

        if self.Xinit is not None: 
            self.optimize_gp()
        else:
            self.generate_initial_design(num_init)
            self.optimize_gp()
        self.fxmean, self.fxvar = self.model.predict(self.Xinit)

        #max., argmax., EI at argmax. 
        self.fxopt = np.max(self.fxmean)
        self.xopt = self.Xinit[np.argmax(self.fxmean)][None, :]  #<--  shape 1 x self.dim
        self.best_ei = self.ei(self.xopt) 
        self.ei_history = [self.ei(x[None,:]) for x in self.Xinit]
        self.grid = latin_center(ei_opt_points, self.dim)
        for i in xrange(self.dim):
            a = self.bounds[i][0]
            b = self.bounds[i][1]
            self.grid[:, i] = a + (b-a) * self.grid[:, i]

    #define the expected improvement function 
    def ei(self, x):
        """
        Evaluate the expected improvement at the given location. 
        """
        if type(x) is not np.ndarray:
            if type(x) is list or type(x) is tuple:
                x = np.array(x)[:, None]
            else:
                x = np.array([[x]])
        else:
            if x.ndim == 1:
                x = x[:, None]

        mu, sigma2 = self.model.predict(x)   #this predicts the mean and variance at x based on current GP 
        sigma = np.sqrt(sigma2)              
        Z = (mu[0, 0] - self.fxopt) / sigma[0, 0]
        pdf = norm.pdf(Z)
        cdf = norm.cdf(Z)
        return ((mu[0, 0]-self.fxopt)*cdf + sigma*pdf)[0, 0]

    def augment_data(self, x, y):
        """
        Augment the available dataset. 
        """
        assert x.ndim == 2 and y.ndim == 2
        self.Xinit = np.vstack([self.Xinit, x])
        self.Yinit = np.vstack([self.Yinit, y])

    def optimize_gp(self):
        """
        Re-initialize the GPR model and optimize hyperparameters. 
        """
        self.model = GPRegression(self.Xinit, self.Yinit, self.kernel)
        self.model.optimize_restarts(self.num_restarts)

    def optimize(self):
        #loop over the maximum number of iterations. 
        for i in xrange(self.maxiter):
            #self.optimize_one_step()
            print "BGO Iteration : "+str(i+1)
            ei_grid = np.array([self.ei(grid_loc[None, :]) for grid_loc in self.grid])
            idx = np.argmax(ei_grid)
            ei_max = np.max(ei_grid)
            x_next = self.grid[idx][None, :]  # shape <- 1 \times self.ndim
            y_next = np.atleast_2d(self.objfunc(x_next[0]))
            self.augment_data(x_next, y_next)
            self.optimize_gp()
            self.mean = self.model.predict(x_next)[0]
            self.fxopt = self.mean[0, 0]
            self.xopt = x_next[0]
            self.ei_history.append(ei_max)
            if self.verbose:
                print "-"*40
                print "Current state  of GP surrogate : "
                print "-"*40
                print self.model
            if self.plot:
                self.make_plot(i)

    
    def make_plot(self, i):
        #plot if one dimensional input
        if self.dim == 1:
            x = np.linspace(self.bounds[0][0], self.bounds[0][1], 200)
            yp, yv = self.model.predict(x[:, None])
            ysd =  np.sqrt(yv)
            yp = yp[:, 0]
            ysd = ysd[:, 0]
            z = np.array([self.ei(grid_loc) for grid_loc in x])

            #plot the current state of the GP surrogate and the current max. 
            plt.plot(x, yp, linewidth=3, label='GP surrogate mean')
            plt.fill_between(x, yp - 2*ysd, yp + 2*ysd, color='blue', alpha=0.2)
            plt.plot(self.xopt[0], self.fxopt, marker = 'D', color = 'red')
            plt.plot(self.Xinit[:, 0], self.Yinit[:, 0], 'x', markersize = 10, color='black', label='data')
            plt.legend(loc='best')
            figname = 'bgo_opt_iter_'+str(i+1)+'.pdf'
            plt.savefig(os.path.join(self.resultdir, figname))
            plt.close()

            #plot the EI func. 
            plt.plot(x, z, 'o')
            figname = 'ei_grid_iter_'+str(i+1)+'.pdf'
            plt.savefig(os.path.join(self.resultdir, figname))
            plt.close()
            
        #plot if 2D input.
        if self.dim == 2:
            x1 = np.linspace(self.bounds[0][0], self.bounds[0][1], 200)
            x2 = np.linspace(self.bounds[1][0], self.bounds[1][1], 200)
            X1, X2 = np.meshgrid(x1, x2)
            X = np.hstack([X1.flatten()[:,None], X2.flatten()[:, None]])
            Z = self.model.predict(X)[0]
            Z = Z.reshape((200, 200))

            #contour plot of GP surrogate mean
            plt.contourf(X1, X2, Z, 100, cmap = 'plasma')
            plt.colorbar()
            plt.tight_layout()
            #plt.scatter([self.xopt[0]], [self.xopt[1]], 'D', label='current optimum')
            #plt.legend(loc='best')
            plt.savefig(os.path.join(self.resultdir, 'bgo_opt_iter_'+str(i+1)+'.pdf'))
            plt.close()

            #contour plot of AF 
            Z = np.array([self.ei(grid_loc[None, :]) for grid_loc in X])
            Z = Z.reshape((200, 200))
            plt.contourf(X1, X2, Z, 100, cmap = 'plasma')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(self.resultdir, 'ei_iter_'+str(i+1)+'.pdf'))
            plt.close()

        if self.dim > 2:
            print "Cannot visualize > 2 dimensions."

    # generate initial design
    def generate_initial_design(self, n):
        """
        :param n: Number of initial samples to generate. 
        """
        #do a LHS sampling to generate input locations 
        self.Xinit = latin_center(n, self.dim)

        for i  in np.arange(self.dim):
            a = self.bounds[i][0]
            b = self.bounds[i][1]
            self.Xinit[:, i] = a + (b-a) * self.Xinit[:, i]
        self.Yinit = np.array([self.objfunc(x) for x in self.Xinit])
        if self.Xinit.ndim < 2:
            self.Xinit = self.Xinit[:, None]
        if self.Yinit.ndim < 2:
            self.Yinit = self.Yinit[:, None]


if __name__ == '__main__':

    def true_func(x):
        true = np.exp(-(x/16.)**2) * np.sin(x / 2.)
        return true 

    def func(x):
        var = 0.01  #at var = 0.2, this optimization breaks down. 
        sd = np.sqrt(var)
        return true_func(x) + sd*np.random.randn()


    def ftrue(x):
        x1 = x[0]
        x2 = x[1]
        true = np.exp(-(x1/16.)**2 - (x2/16)**2) * np.sin(x1/2. + x2/2.)
        return true 

    def f(x):
        var = 0.05
        sd = np.sqrt(var)
        true=ftrue(x)
        return true + sd*np.random.randn()

    #define bounds on the objective function
    bounds = [(-10, 10), (-10, 10)]
    x = np.linspace(-10, 10, 50)
    y = np.linspace(-10, 10, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.ones_like(X)
    for i in xrange(50):
        for j in xrange(50):
            Z[i, j] = ftrue((X[i, j], Y[i, j]))
    plt.contourf(X, Y, Z, cmap = 'plasma')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('objfunc_2d.pdf')
    plt.close()

    #reference solution
    from scipy.optimize import minimize
    res = minimize(ftrue, x0 = np.array([0, 0]))
    xopt_ref = res['x']
    fxopt_ref = res['fun']
    
    #define the optimization problem 
    kernel = GPy.kern.RBF(1, ARD=True)
    bounds = [[-10, 10]]
    optimizer = BayesianOptimizer(objfunc = func, 
                                  bounds = bounds, 
                                  kernel = kernel,
                                  num_init = 20,
                                  num_restarts=1,
                                  verbose=False,
                                  maxiter = 10, 
                                  plot=True,
                                  resultdir = os.path.join(os.getcwd(), 'bgo_res'))
    optimizer.optimize()
    xopt = optimizer.xopt
    fxopt = optimizer.fxopt
    print "BGO x: "+str(xopt)
    print "BGO f(x): "+str(fxopt)

    print "Reference solution :"
    print "xopt = "+str(xopt_ref)
    print "fxopt = "+str(fxopt_ref)
    keyboard()
    exit()
