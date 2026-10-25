"""Render the implemented model, including both residual paths."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
D=Path(__file__).resolve().parent/'docs'
fig,ax=plt.subplots(figsize=(8,11),layout='constrained')
ax.set(xlim=(0,10),ylim=(0,16)); ax.axis('off')
ax.text(5,15.65,'mini-gpt · 14,571,648 parameters',ha='center',fontsize=17,weight='bold')
ax.text(5,15.2,'Decoder-only · pre-norm · 8 blocks · context 128',ha='center',fontsize=11,color='#555555')
def box(y,text,face='#edf3fa'):
    ax.add_patch(FancyBboxPatch((2,y-.34),6,.68,boxstyle='round,pad=.08',facecolor=face,edgecolor='#426382'))
    ax.text(5,y,text,ha='center',va='center',fontsize=10)
def arrow(a,b): ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#426382',lw=1.5))
items=[(14.4,'Token IDs [B, T]'),(13.35,'Token embedding · 513 → 384'),(11.9,'RMSNorm · learned scale · eps 1e-8'),(10.8,'Q / K / V · 6 heads × 64 · no projection biases'),(9.7,'RoPE(Q, K) · base 10000 · V unchanged'),(8.6,'Causal softmax(QKᵀ / √64)V → output projection'),(7.55,'Add attention residual'),(6.5,'RMSNorm · learned scale · eps 1e-8'),(5.45,'Linear 384 → 1536 · ReLU · linear 1536 → 384'),(4.4,'Add FFN residual'),(2.95,'Final RMSNorm'),(1.9,'Untied bias-free LM head · 384 → 513'),(.85,'Logits [B, T, 513] → next-token cross entropy')]
ax.add_patch(FancyBboxPatch((.65,3.82),8.7,8.7,boxstyle='round,pad=.12',facecolor='#f7faf8',edgecolor='#67a585',linestyle='--'))
ax.text(9.05,12.18,'× 8',ha='right',fontsize=12,color='#347153',weight='bold')
for y,t in items: box(y,t)
for (y,_),(z,_) in zip(items,items[1:]): arrow((5,y-.42),(5,z+.42))
ax.plot([5,1.25,1.25],[12.78,12.78,7.55],color='#b5743c',lw=1.5); arrow((1.25,7.55),(1.9,7.55))
ax.text(.95,10,'identity',rotation=90,color='#b5743c',va='center',fontsize=9)
ax.plot([5,8.7,8.7],[7.03,7.03,4.4],color='#b5743c',lw=1.5); arrow((8.7,4.4),(8.1,4.4))
ax.text(8.95,5.8,'identity',rotation=90,color='#b5743c',va='center',fontsize=9)
fig.savefig(D/'architecture.png',dpi=180);fig.savefig(D/'architecture.svg');plt.close(fig)
