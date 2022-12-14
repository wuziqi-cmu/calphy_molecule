"""
calphy: a Python library and command line interface for automated free
energy calculations.

Copyright 2021  (c) Sarath Menon^1, Yury Lysogorskiy^2, Ralf Drautz^2
^1: Max Planck Institut für Eisenforschung, Dusseldorf, Germany 
^2: Ruhr-University Bochum, Bochum, Germany

calphy is published and distributed under the Academic Software License v1.0 (ASL). 
calphy is distributed in the hope that it will be useful for non-commercial academic research, 
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
See the ASL for more details. 

More information about the program can be found in:
Menon, Sarath, Yury Lysogorskiy, Jutta Rogal, and Ralf Drautz.
“Automated Free Energy Calculation from Atomistic Simulations.” Physical Review Materials 5(10), 2021
DOI: 10.1103/PhysRevMaterials.5.103801

For more information contact:
sarath.menon@ruhr-uni-bochum.de/yury.lysogorskiy@icams.rub.de
"""

import numpy as np
import yaml

from calphy.integrators import *
import pyscal.traj_process as ptp
import calphy.lattice as pl
import calphy.helpers as ph
import calphy.phase as cph
from calphy.errors import *

class Liquid(cph.Phase):
    """
    Class for free energy calculation with liquid as the reference state

    Parameters
    ----------
    options : dict
        dict of input options
    
    kernel : int
        the index of the calculation that should be run from
        the list of calculations in the input file

    simfolder : string
        base folder for running calculations

    """
    def __init__(self, options=None, kernel=None, simfolder=None):
        """
        Set up class
        """
        #call base class
        super().__init__(options=options,
        kernel=kernel, simfolder=simfolder)




    def run_averaging(self):
        """
        Run averaging routine

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        Run averaging routine using LAMMPS. Starting from the initial lattice two different routines can
        be followed:
        If pressure is specified, MD simulations are run until the pressure converges within the given
        threshold value.
        If `fix_lattice` option is True, then the input structure is used as it is and the corresponding pressure
        is calculated.
        At the end of the run, the averaged box dimensions are calculated. 
        """
        #create lammps object
        lmp = ph.create_object(self.cores, self.simfolder, self.options["md"]["timestep"], self.options["units"], self.options["atom_style"])

        #set up structure
        lmp = ph.create_structure(lmp, self.calc, self.options["units"], self.options["atom_style"])

        #set up potential
        lmp = ph.set_potential(lmp, self.options)

        #Melt regime for the liquid
        # myRand = 6683
        myRand = np.random.randint(0, 10000)
        print(myRand)
        lmp.velocity("all create", int(self.thigh*.75), myRand)
        lmp.command("fix 0 all shake 1e-6 200 0 b 1 a 1")

        #add some computes
        lmp.command("variable         mvol equal vol")
        lmp.command("variable         mlx equal lx")
        lmp.command("variable         mly equal ly")
        lmp.command("variable         mlz equal lz")
        lmp.command("variable         mpress equal press") 

        #melting cycle
        #try with different multiples of thmult until the structure melts.
        #TODO : Add option to skip melting cycle
        
        melted = False
        
        #this is the multiplier for thigh to try melting routines
        for thmult in np.arange(0.5, 2.0, 0.1):
            
            trajfile = os.path.join(self.simfolder, "traj.melt")
            if os.path.exists(trajfile):
                os.remove(trajfile)

            self.logger.info("Starting melting cycle with thigh temp %f, factor %f"%(self.thigh, thmult))
            lmp.fix("1 all npt temp", self.thigh*thmult, self.thigh*thmult, self.options["md"]["tdamp"], 
                                          "iso", self.p, self.p, self.options["md"]["pdamp"])
            lmp.run(int(self.options["md"]["nsmall"]))
            lmp.unfix(1)
            lmp.dump("2 all custom", 1, trajfile,"id type mass x y z vx vy vz")
            lmp.run(0)
            lmp.undump(2)
            
            #we have to check if the structure melted
            solids = ph.find_solid_fraction(os.path.join(self.simfolder, "traj.melt"))
            self.logger.info("fraction of solids found: %f", solids/self.natoms)
            if (solids/self.natoms < self.options["conv"]["liquid_frac"]):
                melted = True
                break
        
        #if melting cycle is over and still not melted, raise error
        if not melted:
            lmp.close()
            raise SolidifiedError("Liquid system did not melt, maybe try a higher thigh temperature.")

        #now assign correct temperature and equilibrate
        lmp.velocity("all create", self.t, np.random.randint(0, 10000))
        lmp.fix("1 all npt temp", self.t, self.t, self.options["md"]["tdamp"], 
                                      "iso", self.p, self.p, self.options["md"]["pdamp"])
        lmp.run(int(self.options["md"]["nsmall"])) 
        
        #start recording average values
        lmp.command("fix              2 all ave/time %d %d %d v_mlx v_mly v_mlz v_mpress file avg.dat"%(int(self.options["md"]["nevery"]),
            int(self.options["md"]["nrepeat"]), int(self.options["md"]["nevery"]*self.options["md"]["nrepeat"])))

        #monitor the average values until calculation is converged
        laststd = 0.00
        converged = False

        for i in range(self.options["md"]["ncycles"]):
            lmp.command("run              %d"%int(self.options["md"]["nsmall"]))
            ncount = int(self.options["md"]["nsmall"])//int(self.options["md"]["nevery"]*self.options["md"]["nrepeat"])
            #now we can check if it converted
            file = os.path.join(self.simfolder, "avg.dat")
            lx, ly, lz, ipress = np.loadtxt(file, usecols=(1,2,3,4), unpack=True)

            lxpc = ipress
            mean = np.mean(lxpc)
            std = np.std(lxpc)
            volatom = np.mean((lx*ly*lz)/self.natoms)            
            self.logger.info("At count %d mean pressure is %f with vol/atom %f"%(i+1, mean, volatom))

            lmp.command("dump              2 all custom 1 traj.dat id type mass x y z")
            lmp.command("run               0")
            lmp.command("undump            2")
            solids = ph.find_solid_fraction(os.path.join(self.simfolder, "traj.dat"))
            self.logger.info("fraction of solids found: %f", solids/self.natoms)
            if (solids/self.natoms > self.options["conv"]["liquid_frac"]):
                lmp.close()
                raise SolidifiedError('System solidified, increase temperature')


            #check melting;
            if (np.abs(mean - self.p)) < self.options["conv"]["p_tol"]:
                #process other means
                self.lx = np.round(np.mean(lx[-ncount+1:]), decimals=3)
                self.ly = np.round(np.mean(ly[-ncount+1:]), decimals=3)
                self.lz = np.round(np.mean(lz[-ncount+1:]), decimals=3)
                self.volatom = volatom
                self.vol = self.lx*self.ly*self.lz
                self.rho = self.natoms/(self.lx*self.ly*self.lz)

                self.logger.info("finalized vol/atom %f at pressure %f"%(self.volatom, mean))
                self.logger.info("Avg box dimensions x: %f, y: %f, z:%f"%(self.lx, self.ly, self.lz))
                converged = True
                break
            
            laststd = std

        if not converged:
            lmp.close()
            raise ValueError("Pressure did not converge after MD runs, maybe change lattice_constant and try?")

        lmp.command("dump              2 all custom 1 traj.dat id type mass x y z vx vy vz")
        lmp.command("run               0")
        lmp.command("undump            2")
        lmp.command("write_data dump.lmpdata")
        lmp.command("write_restart restart.rst")

        #finish run and close object
        lmp.close()

        #process the trajectory
        self.process_traj()




    def run_integration(self, iteration=1):
        """
        Run integration routine

        Parameters
        ----------
        iteration : int, optional
            iteration number for running independent iterations

        Returns
        -------
        None

        Notes
        -----
        Run the integration routine where the initial and final systems are connected using
        the lambda parameter. See algorithm 4 in publication.
        """
        lmp = ph.create_object(self.cores, self.simfolder, self.options["md"]["timestep"], self.options["units"], self.options["atom_style"])

        lmp.command("read_restart restart.rst")
        lmp.command("fix 0 all shake 1e-6 200 0 b 1 a 1")

        # Adiabatic switching parameters.
        lmp.command("variable        li       equal   1.0")
        lmp.command("variable        lf       equal   0.0")

        #read in the conf file
        conf = os.path.join(self.simfolder, "conf.dump")
        # lmp = ph.read_dump(lmp, conf, species=self.options["nelements"])

        #set hybrid ufm and normal potential
        #lmp = ph.set_hybrid_potential(lmp, self.options, self.eps)
        lmp = ph.set_potential(lmp, self.options)

        #remap the box to get the correct pressure
        lmp = ph.remap_box(lmp, self.lx, self.ly, self.lz)

        
        lmp.command("fix              f1 all nve")
        lmp.command("fix              f2 all langevin %f %f %f %d zero yes"%(self.t, self.t, self.options["md"]["tdamp"], 
                                        np.random.randint(0, 10000)))

        lmp.command("run               %d"%self.options["md"]["te"])

        lmp.command("unfix            f1")
        lmp.command("unfix            f2")

        #---------------------------------------------------------------
        # FWD cycle
        #---------------------------------------------------------------

        lmp.command("variable         flambda equal ramp(${li},${lf})")
        lmp.command("variable         blambda equal 1.0-v_flambda")

        lmp.command("dump dumpxyz all custom 1 tip4p_ufm_traj.dump type x y z")

        lmp.command("pair_style       hybrid/scaled v_flambda %s v_blambda ufm 7.5"%self.options["md"]["pair_style"])

        for pc in self.options["md"]["pair_coeff"]:
            pcraw = pc.split()
            pcnew = " ".join([*pcraw[:2], self.options["md"]["pair_style"].split()[0], *pcraw[2:]])
            lmp.pair_coeff(pcnew)

        # pc =  self.options["md"]["pair_coeff"]
        # pcraw = pc.split()
        # pcnew = " ".join([*pcraw[:2], *[self.options["md"]["pair_style"],], *pcraw[2:]])

        # lmp.command("pair_coeff       %s"%pcnew)
        lmp.command("pair_coeff       * * ufm %f 1.5"%self.eps)

        lmp.command("compute          c1 all pair %s"%self.options["md"]["pair_style"].split()[0])
        lmp.command("compute          c2 all pair ufm")

        lmp.command("variable         step equal step")
        lmp.command("variable         dU1 equal c_c1/atoms")
        lmp.command("variable         dU2 equal c_c2/atoms")

        lmp.command("thermo_style     custom step v_dU1 v_dU2")
        lmp.command("thermo           100")


        # lmp.command("velocity         all create %f %d mom yes rot yes dist gaussian"%(self.t, np.random.randint(0, 10000)))

        lmp.command("fix              f1 all nve")
        lmp.command("fix              f2 all langevin %f %f %f %d zero yes"%(self.t, self.t, self.options["md"]["tdamp"], 
                                        np.random.randint(0, 10000)))
        lmp.command("compute          Tcm all temp/com")
        lmp.command("fix_modify       f2 temp Tcm")

        lmp.command("fix              f3 all print 1 \"${dU1} ${dU2} ${flambda}\" screen no file forward_%d.dat"%iteration)
        lmp.command("run               %d"%self.options["md"]["ts"])

        lmp.command("unfix            f1")
        lmp.command("unfix            f2")
        lmp.command("unfix            f3")
        lmp.command("uncompute        c1")
        lmp.command("uncompute        c2")

        #---------------------------------------------------------------
        # EQBRM cycle
        #---------------------------------------------------------------

        lmp.command("pair_style       ufm 7.5")
        lmp.command("pair_coeff       * * %f 1.5"%self.eps)

        if self.options["md"]["kspace_style"]: lmp.kspace_style("none")

        lmp.command("thermo_style     custom step pe")
        lmp.command("thermo           100")

        lmp.command("fix              f1 all nve")
        lmp.command("fix              f2 all langevin %f %f %f %d zero yes"%(self.t, self.t, self.options["md"]["tdamp"], 
                                        np.random.randint(0, 10000)))
        lmp.command("fix_modify       f2 temp Tcm")

        lmp.command("run               %d"%self.options["md"]["te"])

        lmp.command("unfix            f1")
        lmp.command("unfix            f2")

        #---------------------------------------------------------------
        # BKD cycle
        #---------------------------------------------------------------

        lmp.command("variable         flambda equal ramp(${lf},${li})")
        lmp.command("variable         blambda equal 1.0-v_flambda")

        lmp.command("pair_style       hybrid/scaled v_flambda %s v_blambda ufm 7.5"%self.options["md"]["pair_style"])

        for pc in self.options["md"]["pair_coeff"]:
            pcraw = pc.split()
            pcnew = " ".join([*pcraw[:2], self.options["md"]["pair_style"].split()[0], *pcraw[2:]])
            lmp.pair_coeff(pcnew)
        # lmp.command("pair_coeff       %s"%pcnew)

        if self.options["md"]["kspace_style"]: lmp.kspace_style(self.options["md"]["kspace_style"])

        lmp.command("pair_coeff       * * ufm %f 1.5"%self.eps)

        lmp.command("compute          c1 all pair %s"%self.options["md"]["pair_style"].split()[0])
        lmp.command("compute          c2 all pair ufm")

        lmp.command("variable         step equal step")
        lmp.command("variable         dU1 equal c_c1/atoms")
        lmp.command("variable         dU2 equal c_c2/atoms")

        lmp.command("thermo_style     custom step v_dU1 v_dU2")
        lmp.command("thermo           100")

        lmp.command("fix              f1 all nve")
        lmp.command("fix              f2 all langevin %f %f %f %d zero yes"%(self.t, self.t, self.options["md"]["tdamp"], 
                                        np.random.randint(0, 10000)))
        lmp.command("fix_modify       f2 temp Tcm")

        lmp.command("fix              f3 all print 1 \"${dU1} ${dU2} ${flambda}\" screen no file backward_%d.dat"%iteration)
        lmp.command("run               %d"%self.options["md"]["ts"])

        lmp.command("unfix            f1")
        lmp.command("unfix            f2")
        lmp.command("unfix            f3")
        lmp.command("uncompute        c1")
        lmp.command("uncompute        c2")
        
        #close object
        lmp.close()
    
    def thermodynamic_integration(self):
        """
        Calculate free energy after integration step

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        Calculates the final work, energy dissipation and free energy by
        matching with UFM model
        """
        w, q, qerr = find_w(self.simfolder, nsims=self.nsims, 
            full=True, solid=False)  
        
        #TODO: Hardcoded UFM parameters - enable option to change          
        f1 = get_uhlenbeck_ford_fe(self.t, 
            self.rho, 50, 1.5)

        #Get ideal gas fe
        f2 = get_ideal_gas_fe(self.t, self.rho, 
            self.natoms, self.options["mass"], self.concentration)
        
        self.ferr = qerr
        self.fref = f1
        self.fideal = f2
        self.w = w

        #add pressure contribution if required
        if self.p != 0:
            p = self.p/(10000*160.21766208)
            v = self.vol/self.natoms
            self.pv = p*v
        else:
            self.pv = 0

        #calculate final free energy
        self.fe = self.fideal + self.fref - self.w + self.pv


