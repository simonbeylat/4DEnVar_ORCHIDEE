import numpy as np
import scipy
# from .config import Config
# from .log import Log
# from .output import Output
# from .sites import Sites
# from .atm import Atm

import netCDF4
import warnings
warnings.filterwarnings("ignore")

class FourDEnVar:
    
    def __init__(self,func,xb,xmin,xmax,perr):
        atts = Config.tree.find("minimizer[@id='4DEnVar']").attrib
        self.B_weight=int(atts["B_weight"])
        self.LAS=int(atts["LAS"])
        self.post=int(atts["post"])

        self.func=func
        self.sObs_error=str(atts["Obs_error"])
        print(self.sObs_error)
        self.obs,self.R=self.get_obs()
        self.len_obs=len(self.obs)
        self.xb=xb
        self.xmin=xmin
        self.xmax=xmax
        self.len_xb=len(xb)
        if atts["sample_size"] is 'None':
            self.size=10*self.len_xb
        else:
            self.size=int(atts["sample_size"])
        self.it=int(atts["maxiter"])
        self.Hxb=self.get_prior()
        self.B=self.get_B(perr)
        if Config.recover:
            pass
        else:
            self.ens=self.get_param_ens()
            self.run_ens()

        self.make_hxb()
        if atts["Cost"] is 'Jo':
            self.onlyJo=True
        else:
            self.onlyJo=False

        self.m=int(atts["m"])
        self.maxfun=int(atts["maxfun"])
        self.maxls=int(atts["maxls"])



    def get_obs(self):
        nc = netCDF4.Dataset(Config.path_netcdf)
        if Config.natmsite > 0:
            tmp=nc["data_atm_var0_plume"][0].data
            y=tmp.flatten()
            error=Atm.obs_error.flatten()
        else:
            for isite, site in enumerate(Sites.lot):
                for idx in range(site.nvar):
                    if isite==0 and idx==0:
                        y=nc["data_site%s_var%s" % (site.idx, idx)][0,:].data
                        error=np.ones(len(y))*nc["site_error"][site.idx,idx].data
                    else:
                        tmp_y=nc["data_site%s_var%s" % (site.idx, idx)][0,:].data
                        tmp_error=np.ones(len(tmp_y))*nc["site_error"][site.idx,idx].data
                        y=np.concatenate([y,tmp_y])
                        error=np.concatenate([error,tmp_error])           
        nc.close()
        if self.sObs_error != 'None': error=np.genfromtxt(self.sObs_error)
        R=np.diag(error)
        Log.write(f'Obs = {np.shape(y)}')
        Log.write(f'R = {np.shape(R)}')
        return y,R
    
    def get_prior(self):
        nc = netCDF4.Dataset(Config.path_netcdf)
        if Config.natmsite > 0:
            tmp=nc["data_atm_var0_plume"][1].data
            prior=tmp.flatten()

        else:
            for isite, site in enumerate(Sites.lot):
                for idx in range(site.nvar):
                    if isite==0 and idx==0:
                        prior=nc["data_site%s_var%s" % (site.idx, idx)][1,:].data
                    else:
                        tmp_prior=nc["data_site%s_var%s" % (site.idx, idx)][1,:].data
                        prior=np.concatenate([prior,tmp_prior])                
        nc.close()
        Log.write(f'Prior = {np.shape(prior)}')
        return prior

    def get_B(self,perr):
        B=np.identity(self.len_xb)
        for i in range(self.len_xb):
            B[i,i]=perr[i]**2
        return B


    def run_ens(self):
        X = self.ens.copy() 
        if Config.norc == 1:
            for i in range(len(X)): self.func(X[i,:]) # #
        else:
            indices = np.arange(Config.norc, len(X), Config.norc)
            for subpool in np.split(X, indices): self.func(subpool)
        
    def get_ensemble(self):

        nc = netCDF4.Dataset(Config.path_netcdf)
        totaldata = len(nc.dimensions["data"])
        if Config.recover:
            idx_run=[]
            Config.iter = totaldata
            for i in range(self.size+2):
                if 'iteration' in nc['data_id'][i]:
                    idx_run.append(i)
            id_run=nc['data_id'][idx_run]
        else:
            id_run=nc['data_id'][totaldata-self.size:totaldata]

        id_param=np.zeros(len(id_run))
        for idx,it in enumerate(id_run):
            id_param[idx]=int(it.split(' ')[1])
  
        print(id_param)
        
        if Config.natmsite > 0:
            if Config.recover:
                tmp=nc["data_atm_var0_plume"][idx_run].data
            else:
                tmp=nc["data_atm_var0_plume"][totaldata-self.size:totaldata].data
            tmp_shape=np.shape(tmp)
            ens_computed=tmp.reshape([self.size,tmp_shape[1]*tmp_shape[2]])
        else:
            for isite, site in enumerate(Sites.lot):
                for idx in range(site.nvar):
                    if isite==0 and idx==0:
                        if Config.recover:
                            ens_computed=nc["data_site%s_var%s" % (site.idx, idx)][idx_run,:].data
                        else:
                            ens_computed=nc["data_site%s_var%s" % (site.idx, idx)][totaldata-self.size:totaldata,:].data
                    else:
                        if Config.recover:
                            tmp_ens_computed=nc["data_site%s_var%s" % (site.idx, idx)][idx_run,:].data
                        else:
                            tmp_ens_computed=nc["data_site%s_var%s" % (site.idx, idx)][totaldata-self.size:totaldata,:].data
                        ens_computed=np.concatenate([ens_computed,tmp_ens_computed],axis=1)

        self.ens=nc["param"][id_param.astype(int)].data
        nc.close()
        
        return ens_computed

    def make_hxb(self):
        ens_computed=self.get_ensemble()

        self.X=np.zeros([self.len_xb,self.size])
        self.HX=np.zeros([self.len_obs,self.size])
        for idx,xi in enumerate(self.ens):
            self.X[:,idx]=(1. / (np.sqrt(self.size - 1)))*(xi-self.xb)
            self.HX[:,idx]=(1. / (np.sqrt(self.size - 1)))*(ens_computed[idx,:]-self.Hxb)
        self.X_inv=np.linalg.pinv(self.X)

    def x2w(self,x):
        return np.dot(self.X_inv,(x-self.xb))

    def w2x(self,w):
        return self.xb+np.dot(self.X,w)

    def Jobs(self,w):
        R_inv=np.linalg.inv(self.R)
        delta=np.dot(self.HX, w) + self.Hxb - self.obs
        return np.dot(np.dot(delta.T,R_inv),delta)

    def J(self,w):
        Jo=self.Jobs(w)
        Jb=self.B_weight*np.dot(w, w.T)
        J=0.5*Jo if self.onlyJo else 0.5*(Jo+Jb)
        return J

    def grad(self,w):
        R_inv=np.linalg.inv(self.R)
        delta=np.dot(self.HX, w) + self.Hxb - self.obs
        grad_o=np.dot(np.dot(self.HX.T,R_inv),delta)
        G=grad_o+self.B_weight*w if self.onlyJo else grad_o
        return G
    
    def hess(self,w):
        R_inv=np.linalg.inv(self.R)
        return np.dot(self.HX.T,np.dot(R_inv,self.HX))+np.identity(len(w))+np.eye(self.size)

    def a_cov(self):
        R_inv=np.linalg.inv(self.R)
        return np.linalg.inv(scipy.linalg.sqrtm(np.eye(self.size) + np.dot(self.HX.T,np.dot(R_inv, self.HX))))

    def a_ens(self,xa):
        a_cov_mat=self.a_cov()
        XA=np.dot(self.X,a_cov_mat)
        return np.array([(xa + np.sqrt(self.size-1)*xbi).real for xbi in XA.T]).T

    def find_min_ens_inc(self,x=None,hess=False):
        Log.write("Start minimization of w")
        if x is None:
            x=self.xb
        w = self.x2w(x)
        if hess:
            find_min = scipy.optimize.fmin_ncg(self.J, w, fprime=self.grad,fhess=self.hess, full_output=1,avextol=1e-15)
        else:
            find_min = scipy.optimize.fmin_l_bfgs_b(func=self.J, x0=w, fprime=self.grad,pgtol=1e-15,factr=10,m=self.m,maxfun=self.maxfun,maxls=self.maxls,approx_grad=False,iprint=-1)
        xa = self.w2x(find_min[0])
        return find_min, xa
    
    def callback(self,w):
        x_it=self.w2x(w)
        for idx,xi in enumerate(x_it):
            if xi<self.xmin[idx]:
                Log.write(xi)
                x_it[idx]=self.xmin[idx]
            elif xi>self.xmax[idx]:
                Log.write(xi)
                x_it[idx]=self.xmax[idx]
        Log.write(x_it)
        return self.x2w(x_it)

    def test_bnds(self, xbi):
        
        for idx,xi in enumerate(xbi):
            if self.xmin[idx] <= xi <= self.xmax[idx]:
                continue
            else:
                return True
            
        return False
    
    def generate_param_ens(self,x=None):
        if x is None:
            x=self.xb
        Ensemble=np.random.multivariate_normal(x, self.B, size=self.size)
        for idx,xi in enumerate(Ensemble):
            while self.test_bnds(xi):
                xi = np.random.multivariate_normal(x, self.B)
            Ensemble[idx,:]=xi
        return Ensemble   
    
    def get_param_ens(self,x=None,hypercube=False,usecheck=False):
        Log.write('Generate Ensemble')
        if x is None:
            x=self.xb

        if usecheck:
            X=np.zeros([self.len_xb,self.size])
            while self.check_gen(X):
                Ensemble=self.generate_param_ens(x)
                for idx,xi in enumerate(Ensemble):
                    X[:,idx]=(1. / (np.sqrt(self.size - 1)))*(xi-x)

        elif hypercube:
            import itertools
            diag=np.diag(self.B)
            dxmin=x-diag
            dxmax=x+diag
            Ensemble=np.array(list(itertools.product(*zip(dxmin,dxmax))))
            self.size=len(Ensemble)
        else:
            Ensemble=self.generate_param_ens(x)
        Log.write('Ensemble Generated')
        return Ensemble

    def check_gen(self,X,alpha=0.15):
        std=np.diag(self.B)
        per=(std-np.diag(np.dot(X,X.T)))/std
        for  idx,di in  enumerate(per):
            if -alpha<di<alpha:
                continue
            else:
                return True
        return False

    def update_state(self,xa):
        self.xb=xa
        self.func(xa) 
        nc = netCDF4.Dataset(Config.path_netcdf)
        if Config.natmsite > 0:
            tmp=nc["data_atm_var0_plume"][-1].data
            prior=tmp.flatten()
        else:
            for isite, site in enumerate(Sites.lot):
                for idx in range(site.nvar):
                    if isite==0 and idx==0:
                        prior=nc["data_site%s_var%s" % (site.idx, idx)][-1,:].data
                    else:
                        tmp_prior=nc["data_site%s_var%s" % (site.idx, idx)][-1,:].data
                        prior=np.concatenate([prior,tmp_prior])     
        nc.close()
        self.Hxb=prior
        if self.post:
            XA=self.a_ens(xa)
            self.ens=XA.T
        else:
            self.ens=self.get_param_ens(xa)

        self.run_ens()


    def linear_solution(self,using_R=True):

        delta_y=self.obs - self.Hxb 
    
        if using_R:
            R_inv=np.linalg.inv(self.R)
            KTRinv=np.dot(self.HX.T,R_inv)
            w=np.dot(np.dot(np.linalg.inv(np.dot(KTRinv,self.HX)+self.B_weight*np.eye(self.size)),KTRinv),delta_y)
        else:
            w=np.dot(np.dot(np.linalg.inv(np.dot(self.HX.T,self.HX)),self.HX.T),delta_y)

        return w

    def do_iteration(self,hess=False):
        Log.write("Start iteration")
        for idx in range(1,self.it+1):
            Log.write(f"Iteration: {idx}")
            Log.write(f"Prior J(w)={self.J(self.x2w(self.xb))}")
            if idx == 1:
                if self.LAS:
                    wls=self.linear_solution()
                    xa=self.w2x(wls)
                else:
                    find_min, xa=self.find_min_ens_inc(hess=hess)
            else:
                if self.LAS:
                    wls=self.linear_solution()
                    xa=self.w2x(wls)
                else:
                    find_min, xa=self.find_min_ens_inc(xa,hess=hess)

            if self.it>1:
                Log.write('Update State')
                self.update_state(xa)
                self.make_hxb()

            if self.LAS:
                Log.write(f"Linear Solution J(w) = {self.J(wls)}")
            else:
                Ji=find_min[1]
                Joi=self.Jobs(find_min[0])*0.5
                Log.write(f"Post J(w) = {Ji}")
                Log.write(f"Post Jo(w) = {Joi}")

            if (not hess) & (not self.LAS):
                Log.write(f"find_min = {find_min}")
                if find_min[2]['warnflag']==2:
                    Log.write(find_min[2]['task'])

        return xa
