"""Create reproducible Task 1 evidence figures; never write the assignment report."""
import argparse,csv,json,pathlib,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('--summary',type=pathlib.Path,required=True)
p.add_argument('--output',type=pathlib.Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
with a.summary.open() as f:rows=list(csv.DictReader(f))
if not rows:raise ValueError('No experiment rows')
for xaxis,xlabel,prefix in [
    ('total_packed_bytes','Packed model storage (bytes)','storage'),
    ('eligible_sparsity','Fraction of eligible weights pruned','sparsity')]:
    for ylabel,title in [('wiki_ppl','WikiText-2 PPL'),
                        ('c4_ppl','C4 validation PPL'),
                        ('hs_acc','HellaSwag raw accuracy'),
                        ('hs_acc_norm','HellaSwag char-normalized accuracy')]:
        fig,ax=plt.subplots(figsize=(9,5))
        for row in rows:
            x=float(row[xaxis]);y=float(row[ylabel])
            ax.scatter(x,y,s=45)
            ax.annotate(row['condition'],(x,y),xytext=(3,3),textcoords='offset points',fontsize=7)
        ax.set_xlabel(xlabel);ax.set_ylabel(title);ax.grid(alpha=.3)
        fig.tight_layout();fig.savefig(a.output/f'{prefix}_{ylabel}.png',dpi=180)
        plt.close(fig)
for filename in sorted(a.summary.parent.glob('*_projection_map.json')):
    method=filename.name.split('_projection_map.json')[0]
    obj=json.loads(filename.read_text())
    names=sorted(obj)
    layers=sorted({int(m.group(1)) for name in names if (m:=re.search(r'layers\.(\d+)\.',name))})
    if not layers:continue
    projections=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
    for value,label in [('calibration_output_mse_sampled','Output-error MSE'),
                        ('calibration_output_mse_proxy','Diagonal output-error proxy'),
                        ('zero_weights','Eligible zero weights')]:
        values=np.full((len(layers),len(projections)),np.nan)
        for name,record in obj.items():
            m=re.search(r'layers\.(\d+)\.',name)
            proj=name.rsplit('.',1)[-1]
            if m and proj in projections and value in record:
                values[layers.index(int(m.group(1))),projections.index(proj)]=record[value]
        fig,ax=plt.subplots(figsize=(10,7))
        im=ax.imshow(values,aspect='auto',interpolation='none')
        ax.set_yticks(range(len(layers)),layers)
        ax.set_xticks(range(len(projections)),projections,rotation=45,ha='right')
        ax.set_xlabel('Projection');ax.set_ylabel('Transformer block')
        ax.set_title(f'{method}: {label}')
        fig.colorbar(im,ax=ax)
        fig.tight_layout();fig.savefig(a.output/f'{method}_{value}.png',dpi=180)
        plt.close(fig)
print('Wrote Task 1 evidence figures to',a.output)
