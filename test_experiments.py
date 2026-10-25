import unittest
import torch
from model import GPT
from generate import generate
from experiments import BASE, MODEL_KEYS, VARIANTS, batch

class Checks(unittest.TestCase):
    def test_variants(self):
        original = None
        for name, changes in VARIANTS.items():
            c = dict(BASE, **changes)
            c.update(d_model=16, n_heads=2, d_ff=32, n_layers=2)
            torch.manual_seed(42)
            m = GPT(**{k:c[k] for k in MODEL_KEYS})
            weights = {k:v for k,v in m.state_dict().items()}
            if original is None: original = weights
            for k,v in weights.items(): self.assertTrue(torch.equal(v,original[k]), k)
            x = torch.randint(0, 513, (2, 6))
            z = x.clone(); z[:, -1] = (z[:, -1]+1)%513
            a,b = m(x),m(z)
            self.assertTrue(torch.allclose(a[:,:-1], b[:,:-1],atol=1e-6), name)
            a.mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
            self.assertEqual(m.blocks[0].attention.use_rope,c['use_rope'])
            self.assertEqual(m.blocks[0].ffn.activation,c['activation'])
    def test_sampling(self):
        c = dict(BASE)
        data = torch.arange(1000)
        a = batch(data,c,torch.Generator().manual_seed(43),'cpu')
        b = batch(data,c,torch.Generator().manual_seed(43),'cpu')
        self.assertTrue(torch.equal(a[0],b[0]))
        self.assertTrue(torch.equal(a[0][:,1:],a[1][:,:-1]))
    def test_eos_stops_immediately(self):
        class EOS(torch.nn.Module):
            def __init__(self): super().__init__(); self.calls=0
            def forward(self,x):
                self.calls+=1
                v=torch.full((*x.shape,3),float('-inf')); v[...,2]=0
                return v
        m=EOS(); x=torch.tensor([[0,1]])
        y=generate(m,x,100,16,eos_id=2)
        self.assertEqual(m.calls,1)
        self.assertEqual(y.tolist(),[[0,1,2]])
if __name__ == '__main__': unittest.main()
