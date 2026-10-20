"""Original-dimension Zhang2018 PIN/metal unit cell, COMSOL 6.4.

Reference control without bias lines (shown in source SI9b/c); ideal PEC
zero-thickness metallization and PEC via, lossy source F4B. Capacitance
ambiguity is preserved as two separately named cases, never silently fixed.
Run in an environment providing MPh and JPype, connected to COMSOL 6.4.
"""
import os
os.environ['PYTHONUTF8']='1'
from pathlib import Path
import argparse,json,time,traceback
import numpy as np
import mph
from jpype import JInt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'sptc_toolbox/results/zhang2018_original_alignment_20261019/unitcell'
MODELS=ROOT/'.sors-runs/zhang2018-models'

def save_json(path,data):
    def plain(x):
        if isinstance(x,np.ndarray):return plain(x.tolist())
        if isinstance(x,np.generic):return plain(x.item())
        if isinstance(x,complex):return [x.real,x.imag]
        if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [plain(v) for v in x]
        return x
    Path(path).write_text(json.dumps(plain(data),indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')

def box(c,tag,dim,x,y,z):
    s=c.selection().create(tag,'Box');s.set('entitydim',JInt(dim));s.set('condition','inside')
    for name,bounds in zip('xyz',[x,y,z]):
        s.set(name+'min',str(bounds[0])+'[mm]');s.set(name+'max',str(bounds[1])+'[mm]')
    return s

def build(client,args):
    model=client.create('Zhang2018_original_unitcell')
    j=model.java;c=j.component().create('comp1',True);g=c.geom().create('geom1',3);g.lengthUnit('mm')
    for key,value in {'Rpin':'7.8[ohm]','Lpin':'30[pH]','Cpin':'28[pF]','diode_on':'1'}.items():j.param().set(key,value)
    c.variable().create('circuit').set('Zpin','diode_on*Rpin+(1-diode_on)/(i*2*pi*freq*Cpin)+i*2*pi*freq*Lpin')
    for tag,pos,size in [('air',[-7.5,-7.5,-5],[15,15,17]),('sub',[-7.5,-7.5,0],[15,15,2])]:
        b=g.create(tag,'Block');b.set('pos',pos);b.set('size',size);b.set('selresult',True)
    via=g.create('via','Cylinder');via.set('pos',[-5.35,0.,0.]);via.set('r',.15);via.set('h',2.);via.set('selresult',True)
    wp=g.create('metal','WorkPlane');wp.set('quickz','2');wp.set('unite',True);wp.set('selresult',True)
    wg=wp.geom()
    for tag,pos,size in [('patch',[-3.8,-3.],[7.6,6.]),('rstub',[-4.3,-.15],[.5,.3]),
                         ('lstub',[-5.35,-.15],[.75,.3])]:
        r=wg.create(tag,'Rectangle');r.set('pos',pos);r.set('size',size)
    pad=wg.create('pad','Circle');pad.set('r',.35);pad.set('pos',[-5.35,0.])
    metal_inputs=['patch','rstub','lstub','pad']
    if args.bias_line:
        line=wg.create('bias','Rectangle');line.set('pos',[-.1,-7.5]);line.set('size',[.2,15.]);metal_inputs.append('bias')
    union=wg.create('union','Union');union.selection('input').set(metal_inputs);union.set('intbnd',False)
    wp2=g.create('gap','WorkPlane');wp2.set('quickz','2');wp2.set('unite',True);wp2.set('selresult',True)
    gap=wp2.geom().create('rect','Rectangle');gap.set('pos',[-4.6,-.15]);gap.set('size',[.3,.3])
    g.run()
    print('Geometry built',flush=True)
    # All selections derive from geometric features/coordinate rules.
    box(c,'top',2,[-7.501,7.501],[-7.501,7.501],[11.999,12.001])
    box(c,'bottom',2,[-7.501,7.501],[-7.501,7.501],[-5.001,-4.999])
    box(c,'ground',2,[-7.501,7.501],[-7.501,7.501],[-.001,.001])
    for name,axis,pos in [('xm','x',-7.5),('xp','x',7.5),('ym','y',-7.5),('yp','y',7.5)]:
        bounds=[[-7.501,7.501],[-7.501,7.501],[-5.001,12.001]]
        bounds['xyz'.index(axis)]=[pos-.001,pos+.001];box(c,name,2,*bounds)
    box(c,'substrate',3,[-7.501,7.501],[-7.501,7.501],[-.001,2.001])
    box(c,'viaboundary',2,[-5.501,-5.199],[-.151,.151],[-.001,2.001])
    dom=list(range(1,int(g.getNDomains())+1));via_dom=[int(x) for x in c.selection('geom1_via_dom').entities()]
    domains=[x for x in dom if x not in via_dom]
    emw=c.physics().create('emw','ElectromagneticWaves','geom1');emw.selection().set(domains)
    for axis in ('x','y'):
        pc=emw.create('pc'+axis,'PeriodicCondition',2)
        ids=[int(x) for side in ('m','p') for x in c.selection(axis+side).entities()]
        pc.selection().set(ids);pc.set('PeriodicType','Floquet');pc.set('Floquet_source','UserDefined');pc.set('kFloquet',['0','0','0'])
    for num,side in [(1,'top'),(2,'bottom')]:
        port=emw.create('port'+str(num),'Port',2);port.selection().named(side)
        port.set('PortType','Periodic');port.set('PortName',str(num));port.set('PortExcitation','on' if num==1 else 'off')
        port.set('Polarization','UserDefined');port.set('InputType','E');port.set('Eampl',['1','0','0']);port.set('alpha1_inc','0')
    metal_ids=set(int(x) for x in c.selection('geom1_metal_bnd').entities())
    metal_ids.update(int(x) for x in c.selection('ground').entities())
    metal_ids.update(int(x) for x in c.selection('viaboundary').entities())
    gap_ids=set(int(x) for x in c.selection('geom1_gap_bnd').entities())
    metal_ids-=gap_ids
    pec=emw.create('metalpec','PerfectElectricConductor',2);pec.selection().set(sorted(metal_ids))
    le=emw.create('pin','LumpedElement',2);le.selection().named('geom1_gap_bnd')
    le.set('PortType','UserDefined');le.set('hPort','.3[mm]');le.set('wPort','.3[mm]');le.set('ahPort',['1','0','0'])
    le.set('LumpedElementType','UserDefinedImpedance');le.set('Zelement','Zpin')
    air=c.material().create('air','Common');air.selection().all()
    for prop,value in [('relpermittivity','1'),('relpermeability','1'),('electricconductivity','0')]:air.propertyGroup('def').set(prop,value)
    sub=c.material().create('f4b','Common');sub.selection().named('substrate')
    for prop,value in [('relpermittivity','2.65*(1-i*0.001)'),('relpermeability','1'),('electricconductivity','0')]:sub.propertyGroup('def').set(prop,value)
    mesh=c.mesh().create('mesh1');mesh.autoMeshSize(args.mesh)
    mesh.run()
    print('Mesh built',flush=True)
    study=j.study().create('std1');freq=study.create('freq','Frequency');freq.set('plist','10[GHz]')
    select={str(tag):[int(x) for x in c.selection(str(tag)).entities()] for tag in c.selection().tags() if not str(tag).endswith('_pnt')}
    meta=dict(geometry_domains=dom,emw_domains=domains,via_domains=via_dom,selections=select,
              metal_boundary_ids=sorted(metal_ids),gap_boundary_ids=sorted(gap_ids),mesh_size=args.mesh,
              scope='SI9 geometry; ideal zero-thickness PEC and PEC via, lossy stated F4B; no parameter fit',
              bias_line=args.bias_line,bias_assumption='Continuous .2mm line, no undocumented choke placement' if args.bias_line else 'Source SI9 no-bias control',
              air_top_z_mm=12,substrate_top_z_mm=2,reflection_reference='S11 at top z12; surface-reference factor computed explicitly')
    save_json(OUT/f'geometry_mesh{args.mesh}.json',meta)
    return model

def solve(model,args):
    rows=[]
    cases=[('ON_R7p8',1,'28[pF]'),('OFF_literal_28pF',0,'28[pF]'),('OFF_control_0p028pF',0,'0.028[pF]')]
    for label,on,cap in cases:
        model.parameter('diode_on',str(on));model.parameter('Cpin',cap)
        for f in args.frequencies:
            model.java.study('std1').feature('freq').set('plist',f'{f}[GHz]')
            start=time.perf_counter()
            model.java.study('std1').run()
            s11=complex(np.asarray(model.evaluate('emw.S11')).squeeze())
            s21=complex(np.asarray(model.evaluate('emw.S21')).squeeze())
            # COMSOL exp(+iwt): propagation from z=12 to the z=2 surface and
            # back is exp(-2ikL). Independent ground-only check fixes sign.
            surface=s11*np.exp(2j*2*np.pi*f*1e9/299792458*.010)
            rows.append(dict(case=label,frequency_GHz=f,diode_on=on,capacitance=cap,
                S11=s11,S21=s21,surface_reflection=surface,amplitude=abs(s11),
                phase_surface_deg=float(np.angle(surface,deg=True)),absorption=1-abs(s11)**2-abs(s21)**2,
                seconds=time.perf_counter()-start))
            save_json(OUT/f'sweep_mesh{args.mesh}.json',dict(rows=rows,completed=False,mesh=args.mesh))
            print(json.dumps({k:rows[-1][k] for k in ['case','frequency_GHz','amplitude','phase_surface_deg','seconds']}),flush=True)
    save_json(OUT/f'sweep_mesh{args.mesh}.json',dict(rows=rows,completed=True,mesh=args.mesh))
    model.save(str(MODELS/f'zhang2018_unitcell_mesh{args.mesh}_solved.mph'))
    # Independently known normal-incidence grounded dielectric slab. Restore
    # homogeneous substrate in the via domain and make all top patterns
    # transparent; keep the actual periodic ports and mesh for the check.
    c=model.java.component('comp1');emw=c.physics('emw')
    emw.selection().all();emw.feature('metalpec').selection().named('ground');emw.feature('pin').active(False)
    model.java.study('std1').feature('freq').set('plist','10[GHz]');model.java.study('std1').run()
    observed=complex(np.asarray(model.evaluate('emw.S11')).squeeze())*np.exp(2j*2*np.pi*10e9/299792458*.010)
    n=np.sqrt(2.65*(1-.001j));r01=(1-n)/(1+n);phase=np.exp(-2j*n*2*np.pi*10e9/299792458*.002)
    expected=(r01-phase)/(1-r01*phase)
    save_json(OUT/f'grounded_slab_null_mesh{args.mesh}.json',dict(observed=observed,analytic=expected,
        absolute_complex_error=abs(observed-expected),reference_plane='z=2mm',f_GHz=10.,
        scope='Same mesh and periodic ports; PEC grounded dielectric slab analytic reflection'))
    model.save(str(MODELS/f'zhang2018_grounded_slab_control_mesh{args.mesh}.mph'))
    return rows

def main():
    global OUT,MODELS
    p=argparse.ArgumentParser();p.add_argument('--mesh',type=int,default=6);p.add_argument('--solve',action='store_true')
    p.add_argument('--bias-line',action='store_true')
    p.add_argument('--models',type=Path,default=MODELS,help='Directory for generated COMSOL models')
    p.add_argument('--output',type=Path,default=OUT,help='Directory for numerical summaries')
    p.add_argument('--load',type=Path);p.add_argument('--frequencies',type=float,nargs='+',default=[10.])
    args=p.parse_args()
    MODELS=args.models;OUT=args.output
    if args.bias_line:OUT=OUT/'bias_line';MODELS=MODELS/'bias_line'
    OUT.mkdir(parents=True,exist_ok=True);MODELS.mkdir(parents=True,exist_ok=True)
    client=mph.start(cores=4,version='6.4')
    try:
        model=client.load(str(args.load)) if args.load else build(client,args)
        path=MODELS/f'zhang2018_unitcell_mesh{args.mesh}_built.mph';model.save(str(path))
        print('Saved',str(path).encode('ascii','backslashreplace').decode(),flush=True)
        if args.solve:solve(model,args)
        save_json(OUT/f'run_mesh{args.mesh}.json',dict(completed=True,solved=args.solve,model=str(path),
            parameters=dict(model.parameters()),physics=model.physics(),studies=model.studies(),problems=str(model.problems())))
    except Exception as exc:
        text=traceback.format_exc();(OUT/f'failure_mesh{args.mesh}_{time.time_ns()}.txt').write_text(text,encoding='utf8');print(text,flush=True);raise

if __name__=='__main__':main()
