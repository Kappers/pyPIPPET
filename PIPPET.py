'''
PIPPET: Phase Inference from Point Process Event Timing [1]
    +   mPIPPET: multiple event streams [1]
    +   pPIPPET: pattern inference [2]
    + oscPIPPET: oscillatory PIPPET

Python variant of Jonathan Cannon's original MATLAB implementation:
    https://github.com/joncannon/PIPPET


TODO:
- Surprisal/Gradients for pPIPPET
- Refactor for mpPIPPET, multi-stream per template
- poscPIPPET, because, why not?

[1] Expectancy-based rhythmic entrainment as continuous Bayesian inference.
    Cannon J (2021)  PLOS Computational Biology 17(6): e1009025.
[2] Modeling enculturated bias in entrainment to rhythmic patterns.
    Kaplan T, Cannon J, Jamone L & Pearce M (2021) - In Review.

@Tom Kaplan: t.m.kaplan@qmul.ac.uk
'''
from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from scipy.stats import norm

TWO_PI = 2 * np.pi

@dataclass(init=True, repr=True)
class TemplateParams:
    ''' Expectation template and respective (stimulus) event timing '''
    e_times: np.ndarray   # Observed event times
    e_means: np.ndarray   # Expected event times (mean phase)
    e_vars: np.ndarray    # Variance of expected event times
    e_lambdas: np.ndarray # Strength of expected event times
    label: str            # Identifier/label for analysis

    def reset(self, means: np.ndarray, vars_: np.ndarray, lambdas: np.ndarray) -> None:
        self.e_means = means
        self.e_vars = vars_
        self.e_lambdas = lambdas

@dataclass(init=True, repr=True)
class PIPPETParams:
    ''' Configuration for PIPPET model - parameters and expectation templates '''
    templates: list = field(default_factory=list)

    lambda_0: float = 0.01    # Background event expectation strength
    mu_0: float = 0.0         # Initial estimated phase
    V_0: float = 0.0002       # Initial variance
    sigma_phi: float = 0.05   # Generative model phase noise

    eta_mu: float = 0.0       # Internal phase noise
    eta_V: float = 0.0        # Internal variance noise
    eta_e: float = 0.0        # Internal event noise
    eta_e_share: bool = False # Shared event noise across templates (for pPIPPET, set True)

    dt: float = 0.001         # Integration time step
    overtime: float = 0.0     # Time buffer for simulation
    t0: float = 0.0           # Starting time for simulation with respect to event times
    tmax: float = np.nan      # Maximum time for simulation (otherwise based on event times)

    tau: float = 1.0   # Tempo-like dial for oscPIPPET

    def add(self, times: np.ndarray, means: np.ndarray, vars_: np.ndarray, lambdas: np.ndarray,
            label: str) -> None:
        ''' Add an expectation template, which corresponds to either:
            (1) a unique event stream for mPIPPET,
            (2) a separate expectation template for pPIPPET
        '''
        self.templates.append(TemplateParams(times, means, vars_, lambdas, label))

class PIPPETStream:
    ''' Variational filtering equations for PIPPET, see Methods of [1] or [2] '''

    def __init__(self, params: TemplateParams, lambda_0: float):
        self.params = params
        self.lambda_0 = lambda_0
        self.e_times_p = params.e_times
        # For oscPIPPET:
        self.M = np.arange(-40, 40+1, 1)
        self.cs = np.empty((self.params.e_means.size, self.M.size), dtype=np.clongdouble)
        self.e_means_ = self.params.e_means.reshape(-1, 1)
        self.e_vars_ = self.params.e_vars.reshape(-1, 1)

    @staticmethod
    def z_mu_V(z: complex) -> tuple[float,float]:
        return np.angle(z), -2*np.log(np.abs(z))

    def mu_i(self, mu: float, V: float) -> float:
        return (mu/V + self.params.e_means/self.params.e_vars)/(1/V + 1/self.params.e_vars)

    def K_i(self, V: float) -> float:
        return 1/(1/V + 1/self.params.e_vars)

    def lambda_i(self, mu: float, V: float) -> float:
        gauss = norm.pdf(mu, loc=self.params.e_means, scale=(self.params.e_vars + V)**0.5)
        return self.params.e_lambdas * gauss

    def lambda_hat(self, mu: float, V:float) -> float:
        return self.lambda_0 + np.sum(self.lambda_i(mu, V))

    def mu_hat(self, mu: float, V: float) -> float:
        mu_hat = self.lambda_0 * mu
        mu_hat += np.sum(self.lambda_i(mu, V) * self.mu_i(mu, V))
        return mu_hat / self.lambda_hat(mu, V)

    def V_hat(self, mu_curr: float, mu_prev: float, V: float) -> float:
        V_hat = self.lambda_0 * (V + (mu_prev-mu_curr)**2)
        a = self.lambda_i(mu_prev, V)
        b = self.K_i(V) + (self.mu_i(mu_prev, V)-mu_curr)**2
        V_hat += np.sum(a * b)
        return V_hat / self.lambda_hat(mu_prev, V)

    def zlambda(self, mu: float, V: float, tau: float) -> float:
        self.cs.real = -(self.M**2) * ((V+self.params.e_vars)/2).reshape(-1, 1)
        self.cs.imag = -self.M*(mu - self.params.e_means).reshape(-1, 1)
        y = np.sum(self.params.e_lambdas*tau/TWO_PI * np.exp(self.cs).real.sum(axis=1))
        return self.lambda_0*tau/TWO_PI + y

    def z_hat(self, mu: float, V: float, blambda: float, tau: float) -> complex:
        self.cs.real = -(V*self.M**2)/2 - (self.e_vars_ * (self.M + 1)**2)/2
        self.cs.imag = -self.M*(mu - self.params.e_means).reshape(-1, 1) + self.e_means_
        z_hat_i = self.params.e_lambdas*tau/TWO_PI * np.exp(self.cs).sum(axis=1)
        y = self.lambda_0*tau/TWO_PI * np.exp(complex(-V/2, mu)) + np.sum(z_hat_i)
        return 1/blambda * y

class PIPPET(ABC):
    ''' Base class for PIPPET inference problems '''

    def __init__(self, params: PIPPETParams):
        self.params = params
        # Create unique streams/patterns for (mp)PIPPET filtering, based on params
        self.streams = []
        self.labels = []
        for p in params.templates:
            self.streams.append(PIPPETStream(p, params.lambda_0))
            self.labels.append(p.label)
        self.n_streams = len(self.streams)
        self.event_n = np.zeros(self.n_streams).astype(int)

        # Pre-compute shared internal noise, if appropriate
        if params.eta_e_share:
            noise = np.random.randn(*self.streams[0].e_times_p.shape) * self.params.eta_e
            for s_i in range(self.n_streams):
                self.streams[s_i].e_times_p += noise
        else:
            for s_i in range(self.n_streams):
                noise = np.random.randn(*self.streams[s_i].e_times_p.shape) * self.params.eta_e
                self.streams[s_i].e_times_p += noise
        # Ensure events (perturbed by noise) don't occur at negative time
        for s_i in range(self.n_streams):
            self.streams[s_i].e_times_p[self.streams[s_i].e_times_p < 0] = 0.0

        # Timing of simulation
        self.tmax = params.tmax if ~np.isnan(params.tmax) else max(s.e_times_p[-1] for s in self.streams)
        self.tmax += params.overtime
        self.ts = np.arange(self.params.t0, self.tmax+self.params.dt, step=self.params.dt)
        self.n_ts = self.ts.shape[0]
        # Initialise sufficient statistics
        self.mu_s = np.zeros(self.n_ts)
        self.mu_s[0] = self.params.mu_0
        self.V_s = np.zeros(self.n_ts)
        self.V_s[0] = self.params.V_0
        self.z_s = np.ones(self.n_ts, dtype=np.clongdouble)
        self.z_s[0] = np.exp(complex(-self.params.V_0/2, self.params.mu_0))
        self.idx_event = set()
        self.event_stream = defaultdict(set)
        # Gradient of Lambda
        self.grad = np.zeros((self.n_ts, self.n_streams))
        # Surprisal
        self.surp = np.zeros((self.n_ts, self.n_streams, 2))

    def is_onset(self, t_prev: float, t: float, s_i: int, stim: bool=True) -> bool:
        ''' Check whether an event is observed on this time-step '''
        evts = self.streams[s_i].e_times_p if stim else self.streams[s_i].params.e_means
        if self.event_n[s_i] < len(evts):
            return t_prev <= evts[self.event_n[s_i]] <= t
        return False

    def add_event(self, s_i: int, event_time: float) -> None:
        ''' Add a new event '''
        if self.streams[s_i].e_times_p.size > 0 and event_time < self.streams[s_i].e_times_p[-1]:
            raise ValueError('Existing observation time exceeds new event time')
        n_event = self.streams[s_i].e_times_p.size
        self.streams[s_i].e_times_p = np.insert(self.streams[s_i].e_times_p, n_event, event_time)

    @abstractmethod
    def step(self) -> tuple[float, float]:
        ''' Posterior update for a time step '''
        mu, V = None, None
        return mu, V

    @abstractmethod
    def run(self) -> None:
        ''' Simulation for entire stimulus (i.e. all time steps) '''
        for i in range(1, self.n_ts):
            pass # At least, this should call self.step()


class mPIPPET(PIPPET):
    ''' PIPPET with multiple event streams '''

    def step(self, t_i: float, mu_prev: float, V_prev: float) -> tuple[float, float]:
        ''' Posterior update for a time step '''

        # Internal phase noise
        noise = np.sqrt(self.params.dt) * self.params.eta_mu * np.random.randn()

        # Sum dmu across event streams
        dmu_sum = 0
        for s_i in range(self.n_streams):
            dmu = self.streams[s_i].lambda_hat(mu_prev, V_prev)
            dmu *= (self.streams[s_i].mu_hat(mu_prev, V_prev) - mu_prev)
            dmu_sum += dmu
        mu = mu_prev + self.params.dt*(1 - dmu_sum) + noise

        # Sum dV across event streams
        dV_sum = 0
        for s_i in range(self.n_streams):
            dV = self.streams[s_i].lambda_hat(mu_prev, V_prev)
            dV *= (self.streams[s_i].V_hat(mu, mu_prev, V_prev) - V_prev)
            dV_sum += dV
        V = V_prev + self.params.dt*(self.params.sigma_phi**2 - dV_sum)

        # Update posterior based on events in any stream
        t_prev, t = self.ts[t_i-1], self.ts[t_i]
        for s_i in range(self.n_streams):
            if self.is_onset(t_prev, t, s_i):
                mu_new = self.streams[s_i].mu_hat(mu, V)
                V = self.streams[s_i].V_hat(mu_new, mu, V)
                mu = mu_new
                self.event_n[s_i] += 1
                self.idx_event.add(t_i)
                self.event_stream[t_i].add(s_i)

                self.surp[t_i, s_i, 0] = -np.log(self.streams[s_i].lambda_hat(mu_prev, V_prev)*self.params.dt)
                self.surp[t_i, s_i, 1] = -np.log(self.streams[s_i].lambda_hat(mu, V)*self.params.dt)
                self.grad[t_i, s_i] =  -np.log(self.streams[s_i].lambda_hat(mu_prev+.01, V_prev)*self.params.dt)
                self.grad[t_i, s_i] +=  np.log(self.streams[s_i].lambda_hat(mu_prev-.01, V_prev)*self.params.dt)
                self.grad[t_i, s_i] /= .02
            else:
                self.surp[t_i, s_i, 0] = -np.log(1-self.streams[s_i].lambda_hat(mu_prev, V_prev)*self.params.dt)
                self.surp[t_i, s_i, 1] = -np.log(1-self.streams[s_i].lambda_hat(mu, V)*self.params.dt)
                self.grad[t_i, s_i] =  -np.log(1-self.streams[s_i].lambda_hat(mu_prev+.01, V_prev)*self.params.dt)
                self.grad[t_i, s_i] +=  np.log(1-self.streams[s_i].lambda_hat(mu_prev-.01, V_prev)*self.params.dt)
                self.grad[t_i, s_i] /= .02

        return mu, V

    def run(self) -> None:
        ''' Step through entire stimulus, tracking sufficient statistics '''
        for i in range(1, self.n_ts):
            mu_prev = self.mu_s[i-1]
            V_prev = self.V_s[i-1]
            mu, V = self.step(i, mu_prev, V_prev)
            self.mu_s[i] = mu
            self.V_s[i] = V


class pPIPPET(PIPPET):
    ''' PIPPET with pattern (i.e. template) inference '''

    def __init__(self, params: PIPPETParams, prior: np.ndarray):
        super().__init__(params)

        # Track likelihoods and big Lambdas per pattern
        self.n_m = self.n_streams
        self.L_s = np.zeros(self.n_ts)
        self.L_ms = np.zeros((self.n_ts, self.n_m))
        self.p_m = np.zeros((self.n_ts, self.n_m))
        self.p_m[0] = prior
        self.p_m[0] = self.p_m[0]/self.p_m[0].sum()

        # Initialise big Lambdas using mu_0 and V_0
        for s_i, m in enumerate(self.streams):
            self.L_ms[0, s_i] = m.lambda_hat(self.mu_s[0], self.V_s[0])
        self.L_s[0] = np.sum(self.p_m[0] * self.L_ms[0])

    def step(self, s_i: int, mu_prev: float, V_prev: float, is_event: bool=False) -> tuple[float, float]:
        ''' Posterior step for a given pattern '''

        noise = np.sqrt(self.params.dt) * self.params.eta_mu * np.random.randn()

        dmu = self.streams[s_i].lambda_hat(mu_prev, V_prev)
        dmu *= (self.streams[s_i].mu_hat(mu_prev, V_prev) - mu_prev)
        mu = mu_prev + self.params.dt*(1 - dmu) + noise

        dV = self.streams[s_i].lambda_hat(mu_prev, V_prev)
        dV *= (self.streams[s_i].V_hat(mu, mu_prev, V_prev) - V_prev)
        V = V_prev + self.params.dt*(self.params.sigma_phi**2 - dV)

        if is_event:
            mu_new = self.streams[s_i].mu_hat(mu, V)
            V = self.streams[s_i].V_hat(mu_new, mu, V)
            mu = mu_new

        return mu, V

    def run(self) -> None:
        ''' Step through entire stimulus, for all patterns '''

        # For each time step
        for i in range(1, self.n_ts):
            lambda_prev = self.L_s[i-1]
            mu_prev = self.mu_s[i-1]
            V_prev = self.V_s[i-1]

            mu_ms = np.zeros(self.n_m)
            V_ms = np.zeros(self.n_m)

            t_prev, t = self.ts[i-1], self.ts[i]

            # For each pattern
            for s_i in range(self.n_m):
                lambda_m_prev = self.L_ms[i-1, s_i]
                prev_p_m = self.p_m[i-1, s_i]

                # Update p_m based on event observations (or absence of them)
                is_event = self.is_onset(t_prev, t, s_i)
                d_p_m = prev_p_m * (lambda_m_prev/lambda_prev - 1)
                if not is_event:
                    d_p_m *= -self.params.dt * lambda_prev
                self.p_m[i, s_i] = prev_p_m + d_p_m

                # Update posterior and lambda_m
                mu_m, V_m = self.step(s_i, mu_prev, V_prev, is_event)
                lambda_m = self.streams[s_i].lambda_hat(mu_m, V_m)

                self.L_ms[i, s_i] = lambda_m
                mu_ms[s_i] = mu_m
                V_ms[s_i] = V_m

                if is_event:
                    self.event_n[s_i] += 1
                    self.idx_event.add(i)
                    self.event_stream[i].add(s_i)

            # Marginalize across patterns
            self.mu_s[i] = np.sum(self.p_m[i] * mu_ms)
            self.L_s[i] = np.sum(self.p_m[i] * self.L_ms[i])
            self.V_s[i] = np.sum(self.p_m[i] * V_ms)
            self.V_s[i] += np.sum(self.p_m[i]*(1 - self.p_m[i])*np.power(mu_ms, 2))
            for m in range(self.n_m):
                for n in range(self.n_m):
                    if m != n:
                        self.V_s[i] -= self.p_m[i,m]*self.p_m[i,n]*mu_ms[m]*mu_ms[n]

class oscPIPPET(PIPPET):
    ''' Oscillatory PIPPET '''

    def __init__(self, params: PIPPETParams):
        super().__init__(params)
        self.z_s = np.ones(self.n_ts, dtype=np.clongdouble)
        self.z_s[0] = np.exp(complex(-self.params.V_0/2, self.params.mu_0))

    def step(self, t_i: float, z_prev: complex, mu_prev: float, V_prev: float) -> complex:
        ''' Posterior update for a time step '''

        dz_sum = 0
        for s_i in range(self.n_streams):
            blambda = self.streams[s_i].zlambda(mu_prev, V_prev, self.params.tau)
            z_hat = self.streams[s_i].z_hat(mu_prev, V_prev, blambda, self.params.tau)
            dz_sum += blambda*(z_hat-z_prev)*self.params.dt

        dz_par  =  -(self.params.sigma_phi**2)/2 * self.params.dt
        dz_perp = self.params.tau * self.params.dt
        z = z_prev * np.exp(1j*dz_perp + dz_par) - dz_sum
        # Alternatively:
        #z = z_prev + z_prev*complex(-(self.params.sigma_phi**2)/2, self.params.tau)*self.params.dt - dz_sum

        mu, V_s = PIPPETStream.z_mu_V(z)

        t_prev, t = self.ts[t_i-1], self.ts[t_i]
        for s_i in range(self.n_streams):
            if self.is_onset(t_prev, t, s_i):
                z = self.streams[s_i].z_hat(mu, V_s, self.streams[s_i].zlambda(mu, V_s, self.params.tau), self.params.tau)
                self.event_n[s_i] += 1
                self.idx_event.add(t_i)
                self.event_stream[t_i].add(s_i)

                self.surp[t_i, s_i, 0] = -np.log(self.streams[s_i].lambda_hat(mu_prev, V_prev)*self.params.dt)
                self.surp[t_i, s_i, 1] = -np.log(self.streams[s_i].lambda_hat(mu, V_s)*self.params.dt)
                self.grad[t_i, s_i] =  -np.log(self.streams[s_i].zlambda(mu_prev+.01, V_prev, self.params.tau)*self.params.dt)
                self.grad[t_i, s_i] +=  np.log(self.streams[s_i].zlambda(mu_prev-.01, V_prev, self.params.tau)*self.params.dt)
                self.grad[t_i, s_i] /= .02
            else:
                self.surp[t_i, s_i, 0] = -np.log(1-self.streams[s_i].lambda_hat(mu_prev, V_prev)*self.params.dt)
                self.surp[t_i, s_i, 1] = -np.log(1-self.streams[s_i].lambda_hat(mu, V_s)*self.params.dt)
                self.grad[t_i, s_i] =  -np.log(1-self.streams[s_i].zlambda(mu_prev+.01, V_prev, self.params.tau)*self.params.dt)
                self.grad[t_i, s_i] +=  np.log(1-self.streams[s_i].zlambda(mu_prev-.01, V_prev, self.params.tau)*self.params.dt)
                self.grad[t_i, s_i] /= .02

        return z

    def run(self) -> None:
        ''' Step through entire stimulus, tracking sufficient statistics '''
        for i in range(1, self.n_ts):
            z_prev = self.z_s[i-1]
            mu_prev = self.mu_s[i-1]
            V_prev = self.V_s[i-1]
            z = self.step(i, z_prev, mu_prev, V_prev)
            mu, V = PIPPETStream.z_mu_V(z)
            # Noise
            mu += np.sqrt(self.params.dt) * self.params.eta_mu * np.random.randn()
            V *= np.exp(np.sqrt(self.params.dt) * self.params.eta_V * np.random.randn())
            # Update
            self.mu_s[i], self.V_s[i] = mu, V
            self.z_s[i] = np.exp(complex(-V/2, mu))


if __name__ == "__main__":
    import pdb
    print('Debugger on - press \'c\' to continue examples, \'q\' to quit')

    # PIPPET parameters, including event times and expectations
    p = PIPPETParams()
    p.overtime = 0.2
    e_times   = np.array([0.5, 1.0])
    e_means   = np.array([0.25, 0.5, 0.75, 1.0])
    e_vars    = np.array([0.0001]).repeat(len(e_means))
    e_lambdas = np.array([0.02]).repeat(len(e_means))
    p.add(e_times, e_means, e_vars, e_lambdas, 'Duple')

    # Run PIPPET (mPIPPET but with one expected event stream)
    m = mPIPPET(p)
    print('Running (m)PIPPET...')
    m.run()
    pdb.set_trace()

    # Run mPIPPET - two expected event streams, Duple/Triple
    e_means   = np.array([0.33, 0.66, 1.0])
    e_vars    = np.array([0.0001]).repeat(len(e_means))
    e_lambdas = np.array([0.02]).repeat(len(e_means))
    p.add(e_times, e_means, e_vars, e_lambdas, 'Triple')
    m = mPIPPET(p)
    print('Running mPIPPET...')
    m.run()
    pdb.set_trace()

    # Run pPIPPET - now have competing event streams, equal prior
    prior = np.array([0.5, 0.5])
    m = pPIPPET(p, prior)
    print('Running pPIPPET...')
    m.run()
    pdb.set_trace()

    # Run oscPIPPET - redefine parameter set of wrapped stream
    p = PIPPETParams()
    p.dt = 0.002
    p.overtime = np.pi/10.
    p.sigma_phi = 0.2
    p.mu_0 = 1
    p.V_0 = 10.0
    p.lambda_0 = 0.001
    e_means   = np.array([0])
    e_times   = np.array([np.pi, 3*np.pi -.3, 5*np.pi])
    e_vars    = np.array([0.005]).repeat(len(e_means))
    e_lambdas = np.array([0.02]).repeat(len(e_means))
    p.add(e_times, e_means, e_vars, e_lambdas, '')
    print('Running oscPIPPET...')
    m = oscPIPPET(p)
    m.run()
    pdb.set_trace()

