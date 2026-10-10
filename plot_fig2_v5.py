import argparse, glob, os, re, statistics as st
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
MODES=[("baseline","Baseline"),("yolo","YOLO"),("geo","Geometry"),("proposed","Proposed")]
def read_runs(folder, mode):
    runs={}
    for p in glob.glob(os.path.join(folder, f"{mode}_*.eval")):
        m=re.search(rf"{mode}_(\d+)\.eval$", os.path.basename(p))
        if not m: continue
        for line in open(p):
            t=line.split()
            if len(t)>=2 and t[0]=="rmse": runs[int(m.group(1))]=float(t[1]); break
    return runs
ap=argparse.ArgumentParser()
ap.add_argument("--root", default=os.path.expanduser("~/ORB_SLAM3/results_v5_gpu"))
ap.add_argument("--seq", default="walking_xyz")
ap.add_argument("--out", default="fig2_ate_boxplot")
a=ap.parse_args()
folder=os.path.join(a.root,a.seq); names=[]; data=[]
print("==", a.seq)
for mode,label in MODES:
    r=read_runs(folder,mode)
    if not r: print(" ",label,"no data"); continue
    v=[r[k] for k in sorted(r)]; o=sorted(r.items(), key=lambda kv: kv[1])
    print(f"  {label:9s} n={len(v)} mean={st.mean(v):.4f} std={st.stdev(v):.4f} median={st.median(v):.4f} median-run={o[(len(o)-1)//2][0]}")
    names.append(label); data.append(v)
plt.rcParams.update({"font.family":"serif","font.size":8,"axes.linewidth":0.6,"pdf.fonttype":42})
fig,ax=plt.subplots(figsize=(3.4,2.3))
ax.boxplot(data,widths=0.5,patch_artist=True,showfliers=False,
  medianprops=dict(color="#0d366b",linewidth=1.5),
  boxprops=dict(facecolor="#cde2fb",edgecolor="#256abf",linewidth=0.8),
  whiskerprops=dict(color="#256abf",linewidth=0.8),capprops=dict(color="#256abf",linewidth=0.8))
for i,v in enumerate(data,1):
    n=len(v); ax.scatter([i-0.12+0.24*k/max(n-1,1) for k in range(n)],sorted(v),s=10,color="#2a78d6",edgecolor="white",linewidth=0.4,zorder=3)
ax.set_yscale("log"); lo=min(map(min,data))*0.8; hi=max(map(max,data))*1.3; ax.set_ylim(lo,hi)
tk=[t for t in [0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0] if lo<=t<=hi]; ax.set_yticks(tk); ax.set_yticklabels([f"{t:g}" for t in tk])
ax.set_ylabel("ATE RMSE [m] (log scale)"); ax.set_xticks(range(1,len(names)+1)); ax.set_xticklabels(names)
ax.grid(axis="y",color="#dddddd",linewidth=0.5); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
fig.tight_layout(pad=0.3); fig.savefig(a.out+".pdf"); fig.savefig(a.out+".png",dpi=300)
print("Saved", a.out+".pdf")
