import reboundx
import rebound
import mesa_reader as mr
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import scipy.interpolate as interp
from astropy import constants as const
from astropy import units as u

msol = const.M_sun.to('g').value
rsol = const.R_sun.to('cm').value
au = const.au.to('cm').value
standard_cgrav = const.G.cgs.value
G = const.G.to('au3/(M_sun yr2)').value
M_jup = const.M_jup.to('M_sun').value
R_jup = const.R_jup.to('AU').value
secyer = u.yr.to(u.s)
aursun = u.R_sun.to(u.au)

MESA_INPUT_1={
    # primary variables that go to REBOUND from binary history
    'm1':'star_1_mass',
    'r1':'star_1_radius',
    'mdot1':'lg_mstar_dot_1',
}
MESA_INPUT_2={
    # secondary and binary variables that go to REBOUND from binary history
    'm2':'star_2_mass',
    'r2':'star_2_radius',
    'a':'binary_separation',
}
MESA_INPUT  = {**MESA_INPUT_1, **MESA_INPUT_2}

MESA_INPUT_PRIMARY={
    # primary star variables that go to REBOUND from star1 history
    'M_pri':'star_mass',
    'log_R_pri':'log_R',
    'W_pri':'surf_avg_omega',
    'log_L_pri':'log_total_angular_momentum',
    'I_pri':'total_moment_of_inertia',
    'M_env_pri':'mass_conv_reg_fortides',
    'DR_env_pri':'thickness_conv_reg_fortides',
    'Renv_middle_pri':'radius_conv_reg_fortides',
    'log_Lum_pri':'log_L',
    'conv_mx1_top_r_pri':'conv_mx1_top_r',
    'conv_mx1_bot_r_pri':'conv_mx1_bot_r',
    'surface_h1_pri':'surface_h1',
    
}
MESA_INPUT_SECONDARY={
    # secondary star variables that go to REBOUND from star2 history
    'M_sec':'star_mass',
    'log_R_sec':'log_R',
    'W_sec':'surf_avg_omega',
    'log_L_sec':'log_total_angular_momentum',
    'I_sec':'total_moment_of_inertia',
    'M_env_sec':'mass_conv_reg_fortides',
    'DR_env_sec':'thickness_conv_reg_fortides',
    'Renv_middle_sec':'radius_conv_reg_fortides',
    'log_Lum_sec':'log_L',
    'conv_mx1_top_r_sec':'conv_mx1_top_r',
    'conv_mx1_bot_r_sec':'conv_mx1_bot_r',
    'surface_h1_sec':'surface_h1',
}

class NBSE():
    def __init__(self,
                 binary_path = None,
                 star1_path = None,
                 star2_path = None,
                 kaps_pri_path = None,
                 kaps_sec_path = None,
                 b = None,
                 s1 = None,
                 s2 = None,
                 do_tides = False,
                 lg_mdot_benchmark = -6,
                 t_start = None,
                 t_end = None,
                 t_before_MT=5e6, 
                 t_after_MT=1e6, 
                 delta_m_max = 1e-5, 
                 integrator = 'Whfast'):
        """
        
        Parameter
        -----
        binary_path: str
            path to MESA binary history
        star1_path: str
            path to MESA star 1 history
        star2_path: str
            path to MESA star 2 history
        kaps_pri_path: str
            path to calcuated primary apsidal motion constant file
        kaps_sec_path: str
            path to calcuated secondary apsidal motion constant file
        b: mesa_reader.MesaData
            MESA binary data
        s1: mesa_reader.MesaData
            MESA star 1 data
        s2: mesa_reader.MesaData
            MESA star 2 data
        do_tides: Boolean
            if true: take into account tides between the stars and the planet 
        lg_mdot_benchmark: int
            minimal log10 mass change rate of the donor star that could be identified as mass transfer phase 
        t_start: double
            the time when the simulation starts, if is None, use t_before_MT and t_after_MT
        t_end: double
            the time when the simulation ends, if is None, use t_before_MT and t_after_MT
        t_before_MT: double
            the time simulation starts before the mass transfer phase
        t_after_MT : double
            the time simulation ends after the mass transfer phase
        delta_m_max: double 
            the maximum increase of m1 within one rescaled step
        integrator: str
            integrator in rebound
        """
        
        if binary_path is None:
            binary_path = 'LOGS/binary_history.data'
        if star1_path is None:
            star1_path = './LOGS1/star1.data'
        if star2_path is None:
            star2_path = './LOGS2/star2.data'
        if kaps_pri_path is None:
            kaps_pri_path = './KAPS/kaps_pri.csv'
        if kaps_sec_path is None:
            kaps_sec_path = './KAPS/kaps_sec.csv'
        if b is None:
            b = self.read_mesa(binary_path)
        if s1 is None:
            s1 = self.read_mesa(star1_path)
        if s2 is None:
            s2 = self.read_mesa(star2_path)
        self.do_tides = do_tides
        self.lg_mdot_benchmark = lg_mdot_benchmark
        self.t_before_MT = t_before_MT
        self.t_after_MT = t_after_MT
        self.delta_m_max = delta_m_max
        self.integrator = integrator
        self.b = b
        self.s1 = s1
        self.s2 =s2
        MT_interval = self.find_mass_trasfer_phase(-6)
        age = self.b.data('age')
        if t_start is None or t_end is None:
            t1 = age[MT_interval[0][0]]-t_before_MT  #[0,max(age))
            t2 = age[MT_interval[1][-1]]+t_after_MT  #[0,max(age))
        else:
            t1 = t_start
            t2 = t_end
        times = self.time_update_mesa(self.time_mesa(t1,t2),'m1',delta_m_max)
        t_initial =  times[0]
        binary_interp = {}
        primary_interp = {}
        secondary_interp = {}
        for key in MESA_INPUT:
            binary_interp[key] = self.interpolate_mesa_binary_history(MESA_INPUT[key])
        for key in MESA_INPUT_PRIMARY:
            primary_interp[key] = self.interpolate_mesa_star_history(MESA_INPUT_PRIMARY[key],1)
        for key in MESA_INPUT_SECONDARY:
            secondary_interp[key] = self.interpolate_mesa_star_history(MESA_INPUT_SECONDARY[key],2)
        self.M_pri_initial = primary_interp['M_pri'](t_initial)
        self.M_sec_initial = secondary_interp['M_sec'](t_initial)
        self.a_initial = binary_interp['a'](t_initial) * aursun        
        self.time_interval = times-t_initial
        self.M_pri_interp = primary_interp['M_pri'](times)
        self.M_sec_interp = secondary_interp['M_sec'](times)
        self.a_interp = binary_interp['a'](times) * aursun
        
        if self.do_tides:
            k_pri = np.loadtxt(kaps_pri_path, delimiter=',', usecols=(0, 1))
            kaps_pri_data, kaps_pri_time = k_pri[:, 0], k_pri[:, 1]
            k_sec = np.loadtxt(kaps_sec_path, delimiter=',', usecols=(0, 1))
            kaps_sec_data, kaps_sec_time = k_sec[:, 0], k_sec[:, 1]
            kaps_pri = interp.interp1d(kaps_pri_time, kaps_pri_data)
            kaps_sec = interp.interp1d(kaps_sec_time, kaps_sec_data) 
            self.R_pri_initial = 10**primary_interp['log_R_pri'](t_initial) * aursun
            self.R_sec_initial = 10**secondary_interp['log_R_sec'](t_initial) * aursun
            self.I_pri_initial = primary_interp['I_pri'](t_initial) / msol / au**2
            self.I_sec_initial = secondary_interp['I_sec'](t_initial) / msol / au**2
            self.W_pri_initial = 10**primary_interp['log_L_pri'](t_initial)/primary_interp['I_pri'](t_initial) * secyer
            self.W_sec_initial = 10**secondary_interp['log_L_sec'](t_initial)/secondary_interp['I_sec'](t_initial) * secyer
            self.M_env_pri_initial = primary_interp['M_env_pri'](t_initial)
            self.M_env_sec_initial = secondary_interp['M_env_sec'](t_initial)
            self.DR_env_pri_initial = primary_interp['DR_env_pri'](t_initial)
            self.DR_env_sec_initial = secondary_interp['DR_env_sec'](t_initial)
            self.Renv_middle_pri_initial = primary_interp['Renv_middle_pri'](t_initial)
            self.Renv_middle_sec_initial = secondary_interp['Renv_middle_sec'](t_initial)
            self.log_Lum_pri_initial = primary_interp['log_Lum_pri'](t_initial)
            self.log_Lum_sec_initial = secondary_interp['log_Lum_sec'](t_initial)
            self.log_R_pri_initial = primary_interp['log_R_pri'](t_initial)
            self.log_R_sec_initial = secondary_interp['log_R_sec'](t_initial)
            self.conv_mx1_top_r_pri_initial = primary_interp['conv_mx1_top_r_pri'](t_initial)
            self.conv_mx1_top_r_sec_initial = secondary_interp['conv_mx1_top_r_sec'](t_initial)
            self.conv_mx1_bot_r_pri_initial = primary_interp['conv_mx1_bot_r_pri'](t_initial)
            self.conv_mx1_bot_r_sec_initial = secondary_interp['conv_mx1_bot_r_sec'](t_initial)
            self.surface_h1_pri_initial = primary_interp['surface_h1_pri'](t_initial)
            self.surface_h1_sec_initial = secondary_interp['surface_h1_sec'](t_initial)            
            self.kaps_pri_initial = kaps_pri(t_initial)
            self.kaps_sec_initial = kaps_sec(t_initial)
            
            self.R_pri_interp = 10**primary_interp['log_R_pri'](times) * aursun
            self.R_sec_interp = 10**secondary_interp['log_R_sec'](times) * aursun
            self.I_pri_interp = primary_interp['I_pri'](times) / msol / au**2
            self.I_sec_interp = secondary_interp['I_sec'](times) / msol / au**2
            self.W_pri_interp = 10**primary_interp['log_L_pri'](times)/primary_interp['I_pri'](times) * secyer
            self.W_sec_interp = 10**secondary_interp['log_L_sec'](times)/secondary_interp['I_sec'](times) * secyer
            self.M_env_pri_interp = primary_interp['M_env_pri'](times)
            self.M_env_sec_interp = secondary_interp['M_env_sec'](times)
            self.DR_env_pri_interp = primary_interp['DR_env_pri'](times)
            self.DR_env_sec_interp = secondary_interp['DR_env_sec'](times)
            self.Renv_middle_pri_interp = primary_interp['Renv_middle_pri'](times)
            self.Renv_middle_sec_interp = secondary_interp['Renv_middle_sec'](times)
            self.log_Lum_pri_interp = primary_interp['log_Lum_pri'](times)
            self.log_Lum_sec_interp = secondary_interp['log_Lum_sec'](times)
            self.log_R_pri_interp = primary_interp['log_R_pri'](times)
            self.log_R_sec_interp = secondary_interp['log_R_sec'](times)
            self.conv_mx1_top_r_pri_interp = primary_interp['conv_mx1_top_r_pri'](times)
            self.conv_mx1_top_r_sec_interp = secondary_interp['conv_mx1_top_r_sec'](times)
            self.conv_mx1_bot_r_pri_interp = primary_interp['conv_mx1_bot_r_pri'](times)
            self.conv_mx1_bot_r_sec_interp = secondary_interp['conv_mx1_bot_r_sec'](times)
            self.surface_h1_pri_interp = primary_interp['surface_h1_pri'](times)
            self.surface_h1_sec_interp = secondary_interp['surface_h1_sec'](times)
            self.kaps_pri_interp = kaps_pri(times)
            self.kaps_sec_interp = kaps_sec(times)
            
        self.sim = self.makesim_binary()
    def read_mesa(self, mesa_file):
        """ Read MESA history files with mesa_reader. """
        
        return mr.MesaData(mesa_file)
    def interpolate_mesa_binary_history(self, key):
        """ Interpolate MESA binary history with 'age'.
        
        Parameter
        -----
        key: str
            variable names in MESA binary_history.data
        
        Return
        -----
        history_interp : scipy.interpolate.interpolate.interp1d
            linearly interpolated varibles 
        """
        
        age = self.b.data('age')
        history = self.b.data(key)
        history_interp = interp.interp1d(age,history)
        return history_interp
    
    def interpolate_mesa_star_history(self, key, idx):
        """ Interpolate MESA star history with 'age'.
        
        Parameter
        -----
        key: str
            variable names in MESA history.data
        idx: int
            star index, 1, 2
        
        Return
        -----
        history_interp : scipy.interpolate.interpolate.interp1d
            linearly interpolated varibles
        """
        
        if idx ==1:
            s =self.s1
        elif idx ==2:
            s =self.s2
        star_age = s.data('star_age')
        history = s.data(key)
        history_interp = interp.interp1d(star_age, history)
        return history_interp
    
    def split_array(self,data):
        """ Split an array into groups with consecutive sequences. """
        return np.split(data, np.where(np.diff(data) != 1)[0]+1)
    
    def find_mass_trasfer_phase(self,lg_mdot_benchmark):
        """ Find intervals of mass transfer phase.
        
        Parameter
        -----
        lg_mdot_benchmark: int
            minimal log10 mass change rate of the donor star that could be identified as mass transfer phase
        
        Return
        -----
        index_group : list
            list of indexes of mass transfer phase
        """
        
        lg_mstar_dot_1 = self.b.data('lg_mstar_dot_1')
        index_mdot = np.where(lg_mstar_dot_1 > lg_mdot_benchmark)
        index_group = self.split_array(index_mdot[0])
        return index_group
    
    def time_mesa(self,t1,t2):
        """ get the MESA time sequence within input time range t1 and t2.
        
        Parameter
        -----
        t1: float
            start time of the simulation
        t2: float
            end time of the simulation
        x1: int
            index of the binary age that is the closest to t1
        x2: int
            index of the binary age that is the closest to t2
            
        Return
        -----
        times: ndarray
            MESA time sequence for the simulation including t1 and t2
        """
        
        age = self.b.data('age')
        x1 = np.searchsorted(age, t1)
        x2 = np.searchsorted(age, t2)-1
        times = age[np.arange(x1,x2)]
        if t1 != age[x1]:
            times = np.insert(times, 0, t1)
        if t2 != age[x2]:
            times = np.insert(times, len(times), t2)
        return times
    
    def time_update_mesa(self, times, key, deltamax):
        """ re-calibrate the time sequence based on the threshold of the change of the input varible.
        
        Parameter
        -----
        times: ndarray
            MESA time sequence before calibrate
        key: str
            the varible that is used to refine the time steps
        deltamax: float
            the maximum change of the variable within one time step
            
        Return
        -----
        np.array(t): ndarray
            calibrated time sequence for the simulation, within each time step the change of the varible is smaller than 'deltamax'
        """
        
        t = []
        if key not in MESA_INPUT:
            raise Exception('key not in MESA_INPUT')
        arr = self.interpolate_mesa_binary_history(MESA_INPUT[key])(times)
        for i in range(len(times) - 1):
            if arr[i]-arr[i+1]> deltamax:
                n = int((arr[i]-arr[i+1])/deltamax + 2)
                t.extend(np.linspace(times[i],times[i+1],n).tolist()[:-1])
            else:
                t.append(times[i])
        t.append(times[-1])
        return np.array(t)
    
    def calculate_kt_conv(self, M_env, DR_env, Renv_middle, Lum, W, M, n):
        """ calculate k/T based on Hurley et al. 2002 for convective envelope
        
        Parameter
        -----
        M_env: float
            mass of the dominant convective region for tides above the core
        DR_env: float
            thickness of the dominant convective region for tides above the core
        Renv_middle : float
            position of the dominant convective region for tides above the core
        Lum: float
            luminosity of the star
        W: float
            average spin angular velocity of the star
        M: float
            total mass of the star
        n: float
            mean orbital angular velocity
            
        Return
        -----
        kT_conv: float
            k/T for convective envelope
        """
        
        tau_conv = 0.431 * ((M_env * DR_env * Renv_middle/ (3 * Lum)) ** (1.0 / 3.0))
        P_spin = 2 * np.pi / W  
        P_orb = 2 * np.pi / n   
        P_tid = np.abs(1 / (1 / P_orb - 1 / P_spin))   
        f_conv = np.min([1, (P_tid / (2 * tau_conv)) ** 2])
        kT_conv = ((2. / 21) * (f_conv / tau_conv) * (M_env / M)) 
        return kT_conv
    def calculate_kt_rad(self, conv_mx1_top_r, conv_mx1_bot_r, R, M1, M2, surface_h1, a): 
        """ calculate k/T based on Hurley et al. 2002, Qin et al. 2018 for radiative envelope
        
        Parameter
        -----
        conv_mx1_top_r: float
            coordinate of top convective mixing zone coordinate in Rsolar
        conv_mx1_bot_r : float
            coordinate of bottom convective mixing zone coordinate in Rsolar
        R: float
            radius of the star
        M1: float
            mass of the target star
        M2: float
            mass of the companion star
        surface_h1: float
            surface mass Hydrogen abundance
        a: float
            orbital semi-major axis
            
        Return
        -----
        kT_conv: float
            k/T for radiative envelope
        """
        
        R_conv = conv_mx1_top_r - conv_mx1_bot_r
        q = M2 / M1
        if (R_conv > R or R_conv <= 0.0 or conv_mx1_bot_r / R > 0.1):
            E2 = 1.592e-9 * M1 ** (2.84)
        else:
            if R <= 0:
                E2 = 0
            elif surface_h1 > 0.4:
                E2 = 10.0 ** (-0.42) * (R_conv / R) ** (7.5)  
            elif surface_h1 <= 0.4:  # "HeStar"
                E2 = 10.0 ** (-0.93) * (R_conv / R) ** (6.7)  
            else:  
                E2 = 1.592e-9 * M1 ** (2.84)      
        kT_rad = np.sqrt(standard_cgrav * (M1 * msol) * (R * rsol)**2 / (a * au)**5) * (1 + q) ** (5.0 / 6) * E2 * secyer
        return kT_rad
        
    def makesim_binary(self):
        """ add a binary in rebound """
        
        sim = rebound.Simulation()
        sim.units = ('yr', 'AU', 'Msun')
        sim.integrator = self.integrator 
        if not self.do_tides:
             sim.add(m=self.M_pri_initial)      
             sim.add(m=self.M_sec_initial,a=self.a_initial)
        else:
            sim.add(m=self.M_pri_initial,r=self.R_pri_initial)      
            sim.add(m=self.M_sec_initial,r=self.R_sec_initial,a=self.a_initial)
        return sim
        
    def add_planet(self, ap_initial, ep_initial, inc_initial, M_jup, R_jup):
        """ add planets in rebound """
        
        self.sim.add(a=ap_initial,e=ep_initial,inc =inc_initial, m=M_jup, r=R_jup)
        self.sim.move_to_com()

    def __call__(self):
                
        if not self.do_tides:
            sim = self.sim
            ps = sim.particles
            Nout = len(self.time_interval)
            a_p = np.zeros(shape=[len(self.sim.particles)-2,Nout])
            e_p = np.zeros(shape=[len(self.sim.particles)-2,Nout])
            inc_p = np.zeros(shape=[len(self.sim.particles)-2,Nout])
            for i, t in enumerate(self.time_interval):
                sim.integrate(t)
                # update MESA quantities
                ps[0].m = self.M_pri_interp[i]
                ps[1].m = self.M_sec_interp[i]
                ps[1].a = self.a_interp[i]
                for k in range(len(self.sim.particles)-2):
                    a_p[k,i] = ps[2+k].a
                    e_p[k,i] = ps[2+k].e
                    inc_p[k,i] = ps[2+k].inc
        else:
            sim = self.sim
            ps = sim.particles
            if len(self.sim.particles) > 3:
                print('Now, the code only supports one planet with tides on')
                return
            Nout = len(self.time_interval)
            a_p = np.zeros(Nout)
            e_p = np.zeros(Nout)
            inc_p = np.zeros(Nout)
            o_p = np.zeros((Nout, 3))
            rebx = reboundx.Extras(sim)
            sf = rebx.load_force("tides_spin")
            rebx.add_force(sf)
            ps[0].params['Omega'] = rebound.spherical_to_xyz(magnitude=self.W_pri_initial, theta=0, phi=0)
            ps[0].params['I'] = self.I_pri_initial
            ps[1].params['Omega'] = rebound.spherical_to_xyz(magnitude=self.W_sec_initial, theta=0, phi=0)
            ps[1].params['I'] = self.I_sec_initial
            ps[2].params['Omega'] = rebound.spherical_to_xyz(magnitude=2*np.pi/(ps[2].P), theta=0, phi=0)
            ps[2].params['I'] = 0.25 * ps[2].m * ps[2].r**2
            ps[0].params['k2'] = 2 * self.kaps_pri_initial
            ps[1].params['k2'] = 2 * self.kaps_sec_initial
            ps[2].params['k2'] = 0.565
            kT_conv_pri_initial = self.calculate_kt_conv(self.M_env_pri_initial, self.DR_env_pri_initial,
                                       self.Renv_middle_pri_initial, 10**self.log_Lum_pri_initial,
                                       self.W_pri_initial, self.M_pri_initial,ps[2].n)
            kT_conv_sec_initial = self.calculate_kt_conv(self.M_env_sec_initial, self.DR_env_sec_initial,
                                       self.Renv_middle_sec_initial, 10**self.log_Lum_sec_initial,
                                       self.W_sec_initial, self.M_sec_initial,ps[2].n)
            kT_rad_pri_initial = self.calculate_kt_rad(self.conv_mx1_top_r_pri_initial, self.conv_mx1_bot_r_pri_initial,
                                       10**self.log_R_pri_initial, self.M_pri_initial, self.M_sec_initial,
                                       self.surface_h1_pri_initial, ps[2].a)
            kT_rad_sec_initial = self.calculate_kt_rad(self.conv_mx1_top_r_sec_initial, self.conv_mx1_bot_r_sec_initial,
                                      10**self.log_R_sec_initial, self.M_sec_initial, self.M_pri_initial,
                                      self.surface_h1_sec_initial, ps[2].a)
            kT_pri_initial = max(kT_conv_pri_initial, kT_rad_pri_initial)
            kT_sec_initial = max(kT_conv_sec_initial, kT_rad_sec_initial)
            # lag time tau is related with T
            ps[0].params['tau'] = kT_pri_initial/self.kaps_pri_initial*(self.R_pri_initial)**3/G/self.M_pri_initial
            ps[1].params['tau'] = kT_sec_initial/self.kaps_sec_initial*(self.R_sec_initial)**3/G/self.M_sec_initial
            # approximation for the planet
            Q = 1e4
            ps[2].params['tau'] = 1/(2*Q*ps[2].n)
            rebx.initialize_spin_ode(sf)
            for i, t in enumerate(self.time_interval):
                sim.integrate(t)
                ps[0].m = self.M_pri_interp[i]
                ps[0].r = self.R_pri_interp[i]
                ps[0].params['I'] = self.I_pri_interp[i]
                ps[0].params['Omega'] = rebound.spherical_to_xyz(magnitude=self.W_pri_interp[i], theta=0, phi=0)
                ps[1].m = self.M_sec_interp[i]
                ps[1].r = self.R_sec_interp[i]
                ps[1].params['I'] = self.I_sec_interp[i]
                ps[1].params['Omega'] = rebound.spherical_to_xyz(magnitude=self.W_sec_interp[i], theta=0, phi=0)
                ps[1].a = self.a_interp[i]
                kT_conv_pri = self.calculate_kt_conv(self.M_env_pri_interp[i],self.DR_env_pri_interp[i],
                                    self.Renv_middle_pri_interp[i],10**self.log_Lum_pri_interp[i],
                                    self.W_pri_interp[i],self.M_pri_interp[i],ps[2].n)
                kT_conv_sec = self.calculate_kt_conv(self.M_env_sec_interp[i],self.DR_env_sec_interp[i],
                                    self.Renv_middle_sec_interp[i],10**self.log_Lum_sec_interp[i],
                                    self.W_sec_interp[i],self.M_sec_interp[i],ps[2].n)
                kT_rad_pri = self.calculate_kt_rad(self.conv_mx1_top_r_pri_interp[i],self.conv_mx1_bot_r_pri_interp[i],
                                  10**self.log_R_pri_interp[i],self.M_pri_interp[i],self.M_sec_interp[i],
                                  self.surface_h1_pri_interp[i],ps[2].a)  
                kT_rad_sec = self.calculate_kt_rad(self.conv_mx1_top_r_sec_interp[i],self.conv_mx1_bot_r_sec_interp[i],
                                  10**self.log_R_sec_interp[i],self.M_sec_interp[i],self.M_pri_interp[i],
                                  self.surface_h1_sec_interp[i],ps[2].a)
                kT_pri = max(kT_conv_pri, kT_rad_pri)
                kT_sec = max(kT_conv_sec, kT_rad_sec)
                ps[0].params['tau'] = kT_pri/self.kaps_pri_interp[i]*(self.R_pri_interp[i])**3/G/self.M_pri_interp[i]
                ps[1].params['tau'] = kT_sec/self.kaps_sec_interp[i]*(self.R_sec_interp[i])**3/G/self.M_sec_interp[i]
                ps[2].params['tau'] = 1/(2*Q*ps[2].n)
                ps[0].params['k2'] = 2*self.kaps_pri_interp[i]
                ps[1].params['k2'] = 2*self.kaps_sec_interp[i]
                a_p[i] = ps[2].a
                e_p[i] = ps[2].e
                inc_p[i] = ps[2].inc
                o_p[i] = ps[2].params['Omega']
        return a_p,e_p,inc_p

CB = NBSE(t_before_MT=1e6,t_after_MT=5e6,do_tides=False)
#Customize simulation time
#CB = NBSE(t_start=5.3242e8,t_end=5.3248e8, do_tides=False)
CB.add_planet(0.5,0.0,0.0,M_jup,R_jup)
orb = CB()
data = np.column_stack((CB.time_interval, orb[0][0],orb[1][0],orb[2][0]))
np.savetxt('orb_without_tides.csv', data, delimiter=',')

### with tides on
CB = NBSE(t_before_MT=1e6,t_after_MT=7.5e6,do_tides=True)
CB.add_planet(0.5,0.0,0.0,M_jup,R_jup)
orb = CB()
# It supports only one planet with tides on due to compuational limit
data = np.column_stack((CB.time_interval, orb[0],orb[1],orb[2]))
np.savetxt('orb_with_tides.csv', data, delimiter=',')
