import numpy as np
import warnings
import scipy
import netCDF4
warnings.filterwarnings("ignore")

class FourDEnVAR:

    def __init__(self,path_netcdf,natmsite,size,B_weight=1,sObs_error="None",onlyJo=False):

        self.path_netcdf=path_netcdf
        self.natmsite=natmsite
        self.size=size
        self.it=1

        self.xb,self.xmin,self.xmax=self.get_xb()

        self.len_xb=len(self.xb)
        self.B_weight=B_weight
        self.sObs_error=sObs_error

        self.obs,self.R=self.get_obs()
        self.len_obs=len(self.obs)
        self.Hxb=self.get_prior()

#         self.B=self.get_B(perr)
#         self.ens=self.get_param_ens_from_path()
        self.make_hxb()
        self.onlyJo=onlyJo


        self.m=50
        self.maxfun=100000
        self.maxls=50

    def get_xb(self):
        nc=netCDF4.Dataset(self.path_netcdf)
        xb=nc['param'][0].data
        xmin=nc['param_min'][:].data
        xmax=nc['param_max'][:].data
        nc.close()
        return xb,xmin,xmax

    def get_obs(self):
        nc = netCDF4.Dataset(self.path_netcdf)
        if self.natmsite > 0:
            tmp=nc["data_atm_var0_plume"][0].data
            y=tmp.flatten()
            error=np.ones(len(y))*0.5
        else:

            pass
#             for isite, site in enumerate(Sites.lot):
#                 for idx in range(site.nvar):
#                     if isite==0 and idx==0:
#                         y=nc["data_site%s_var%s" % (site.idx, idx)][0,:].data
#                         error=np.ones(len(y))*nc["site_error"][site.idx,idx].data
#                     else:
#                         tmp_y=nc["data_site%s_var%s" % (site.idx, idx)][0,:].data
#                         tmp_error=np.ones(len(tmp_y))*nc["site_error"][site.idx,idx].data
#                         y=np.concatenate([y,tmp_y])
#                         error=np.concatenate([error,tmp_error])
        nc.close()
        if self.sObs_error != 'None': error=np.genfromtxt(self.sObs_error)
        R=np.diag(error)
        print(f'Obs = {np.shape(y)}')
        print(f'R = {np.shape(R)}')
        return y,R

    def get_prior(self):
        nc = netCDF4.Dataset(self.path_netcdf)
        if self.natmsite > 0:
            tmp=nc["data_atm_var0_plume"][1].data
            prior=tmp.flatten()

        else:
            pass
#             for isite, site in enumerate(Sites.lot):
#                 for idx in range(site.nvar):
#                     if isite==0 and idx==0:
#                         prior=nc["data_site%s_var%s" % (site.idx, idx)][1,:].data
#                     else:
#                         tmp_prior=nc["data_site%s_var%s" % (site.idx, idx)][1,:].data
#                         prior=np.concatenate([prior,tmp_prior])
        nc.close()
        print(f'Prior = {np.shape(prior)}')
        return prior

#     def get_B(self,perr):
#             perr=nc['param_error'][:]
#         B=np.identity(self.len_xb)
#         for i in range(self.len_xb):
#             B[i,i]=perr[i]**2
#         return B


    def run_ens(self):
#         X = self.ens.copy()
#         if Config.norc == 1:
#             for i in range(len(X)): self.func(X[i,:]) # #
#         else:
#             indices = np.arange(Config.norc, len(X), Config.norc)
#             for subpool in np.split(X, indices): self.func(subpool)

        nc = netCDF4.Dataset(self.path_netcdf)
        id_run=nc['data_id'][2:self.size+2]
        id_param=np.zeros(len(id_run))
#        print(id_run)
        for idx,it in enumerate(id_run):
            id_param[idx]=int(it.split(' ')[1])

        if self.natmsite > 0:
            tmp=nc["data_atm_var0_plume"][2:self.size+2].data
            tmp_shape=np.shape(tmp)
            ens_computed=tmp.reshape([self.size,tmp_shape[1]*tmp_shape[2]])
        else:
#             for isite, site in enumerate(Sites.lot):
#                 for idx in range(site.nvar):
#                     if isite==0 and idx==0:
#                         ens_computed=nc["data_site%s_var%s" % (site.idx, idx)][totaldata-self.size:totaldata,:].data
#                     else:
#                         tmp_ens_computed=nc["data_site%s_var%s" % (site.idx, idx)][totaldata-self.size:totaldata,:].data
#                         ens_computed=np.concatenate([ens_computed,tmp_ens_computed],axis=1)
            pass
        self.ens=nc["param"][id_param.astype(int)].data
        nc.close()

        return ens_computed

    def make_hxb(self):
        ens_computed=self.run_ens()

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
        return np.dot(self.HX.T,np.dot(R_inv,self.HX))+np.eye(self.size)

    def a_cov(self):
        R_inv=np.linalg.inv(self.R)
        return np.linalg.inv(scipy.linalg.sqrtm(np.eye(self.size) + np.dot(self.HX.T,np.dot(R_inv, self.HX))))

    def a_ens(self,xa):
        a_cov_mat=self.a_cov()
        XA=np.dot(self.X,a_cov_mat)
        return np.array([(xa + np.sqrt(self.size-1)*xbi).real for xbi in XA.T]).T

    def find_min_ens_inc(self,x=None,hess=False):
        print("Start minimization of w")
        if x is None:
            x=self.xb
        w = self.x2w(x)
        if hess:
            find_min = scipy.optimize.fmin_ncg(self.J, w, fprime=self.grad,fhess=self.hess, full_output=1,avextol=1e-15)
        else:
            find_min = scipy.optimize.fmin_l_bfgs_b(func=self.J, x0=w, fprime=self.grad,pgtol=1e-15,factr=1e-15,m=self.m,maxfun=self.maxfun,maxls=self.maxls,approx_grad=False,iprint=-1)


        xa = self.w2x(find_min[0])
        return find_min, xa

    def callback(self,w):
        x_it=self.w2x(w)
        for idx,xi in enumerate(x_it):
            if xi<self.xmin[idx]:
                print(xi)
                x_it[idx]=self.xmin[idx]
            elif xi>self.xmax[idx]:
                print(xi)
                x_it[idx]=self.xmax[idx]
        print(x_it)
        return self.x2w(x_it)

    def test_bnds(self, xbi):

        for idx,xi in enumerate(xbi):
            if self.xmin[idx] <= xi <= self.xmax[idx]:
                continue
            else:
                return True

        return False

#     def generate_param_ens(self,x=None):
#         if x is None:
#             x=self.xb
#         Ensemble=np.random.multivariate_normal(x, self.B, size=self.size)
#         for idx,xi in enumerate(Ensemble):
#             while self.test_bnds(xi):
#                 xi = np.random.multivariate_normal(x, self.B)
#             Ensemble[idx,:]=xi
#         return Ensemble

#     def get_param_ens(self,x=None,hypercube=False,usecheck=False):
#         print('Generate Ensemble')
#         if x is None:
#             x=self.xb

#         if usecheck:
#             X=np.zeros([self.len_xb,self.size])
#             while self.check_gen(X):
#                 Ensemble=self.generate_param_ens(x)
#                 for idx,xi in enumerate(Ensemble):
#                     X[:,idx]=(1. / (np.sqrt(self.size - 1)))*(xi-x)

#         elif hypercube:
#             import itertools
#             diag=np.diag(self.B)
#             dxmin=x-diag
#             dxmax=x+diag
#             Ensemble=np.array(list(itertools.product(*zip(dxmin,dxmax))))
#             self.size=len(Ensemble)
#         else:
#             Ensemble=self.generate_param_ens(x)
#         print('Ensemble Generated')
#         return Ensemble

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
#         self.func(xa)
        nc = netCDF4.Dataset(self.path_netcdf)
        if self.natmsite > 0:
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
        XA=self.a_ens(xa)
        #self.ens=self.get_param_ens(xa)
        self.ens=XA.T



    def do_iteration(self,hess=False):
        print("Start iteration")

        for idx in range(1,self.it+1):
            print(f"Iteration: {idx}")
            print(f"Prior J(w)={self.J(self.x2w(self.xb))}")
            if idx == 1:
                find_min, xa=self.find_min_ens_inc(hess=hess)
            else:
#                 print('Update State')
#                 self.update_state(xa)
#                 self.make_hxb()
#                 find_min, xa=self.find_min_ens_inc(xa,hess=hess)
                continue

            print(f"Post J(w) = {find_min[1]}")
#             print(f"check {self.J(find_min[0])}")

#             if not hess:
#                 print(f"find_min = {find_min}")
#                 if find_min[2]['warnflag']==2:
#                     print(find_min[2]['task'])

        return find_min, xa

    def linear_solution(self,using_R=False):

        delta_y=self.obs - self.Hxb

        if using_R:
            R_inv=np.linalg.inv(self.R)
            KTRinv=np.dot(self.HX.T,R_inv)
            w=np.dot(np.dot(np.linalg.inv(np.dot(KTRinv,self.HX)+self.B_weight*np.eye(self.size)),KTRinv),delta_y)
        else:
            w=np.dot(np.dot(np.linalg.inv(np.dot(self.HX.T,self.HX)),self.HX.T),delta_y)

        return w
