"""Same-trajectory DFT accumulation cadence control; no source/field fit."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,json
import numpy as np
import meep as mp
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from meep_smooth_reference import fixture,plane_sources
from meep_floquet_dataset import make_monitors,extract_ports,current_fourier
from meep_yee_green_response import native_points
mp.verbosity(0)
HERE=Path(__file__).resolve().parent
frequencies=[.378,.388,.398,.408,.418,.438]
source=mp.GaussianSource(.398,fwidth=.08,cutoff=7)
sim=fixture(plane_sources(source,1),8,True,1,y_mirror=False)
sim.init_sim()
axes=sim.get_array_metadata(center=mp.Vector3(),size=mp.Vector3(1,1,.75))[:3]
points,_,_=native_points(axes,8)
grid=dict(points=points,resolution=8,dt=sim.fields.dt)
monitors=[sim.add_dft_fields([mp.Ex,mp.Ey,mp.Ez],frequencies,
    center=mp.Vector3(),size=mp.Vector3(.25,.25,.25),yee_grid=True,decimation_factor=d) for d in (1,8)]
port_monitors=[make_monitors(sim,frequencies,grid,d)[1] for d in (1,8)]
rows=[]
for stop in (500,1000,1500):
    sim.run(until=stop-sim.meep_time())
    field=[]
    for m in monitors:
        field.append(np.concatenate([sim.get_dft_array(m,c,f).reshape(-1)
            for c in (mp.Ex,mp.Ey,mp.Ez) for f in range(len(frequencies))]))
    output=[extract_ports(sim,m,frequencies,grid)[0] for m in port_monitors]
    rows.append(dict(time=sim.meep_time(),relative_difference=float(np.linalg.norm(field[1]-field[0])/np.linalg.norm(field[0])),
        per_frequency_output_relative_difference=[float(np.linalg.norm(a-b)/np.linalg.norm(a)) for a,b in zip(output[0],output[1])]))
dt=sim.fields.dt;end=source.swigobj.last_time()
t1=(np.arange(1,int(np.ceil(end/dt))+2)-.5)*dt
j1=np.array([source.swigobj.current(float(t),dt) for t in t1])
fine=current_fourier(source,frequencies,dt)
t8=(np.arange(8,int(np.ceil(end/dt))+9,8)-.5)*dt
j8=np.array([source.swigobj.current(float(t),dt) for t in t8])
coarse=np.exp(2j*np.pi*np.array(frequencies)[:,None]*t8)@j8*(8*dt)/np.sqrt(2*np.pi)
# This source check records only the nominal Nyquist and tail facts;
# actual field DFT1/8 comparison above supplies the decisive control.
summary=dict(status='SAME_STATIC_MAXWELL_TRAJECTORY_DFT_DECIMATION_CONTROL',resolution=8,dt=dt,
    frequencies=frequencies,decimations=[1,8],source=dict(center=.398,fwidth=.08,cutoff=7),rows=rows,
    eight_step_sampling_nyquist=1/(16*dt),source_integrated_current_abs=float(abs(dt*np.sum(j1))),
    per_frequency_source_fft_relative_difference=(abs(coarse-fine)/abs(fine)).tolist())
(HERE/'decimation_control.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True);sim.reset_meep()
