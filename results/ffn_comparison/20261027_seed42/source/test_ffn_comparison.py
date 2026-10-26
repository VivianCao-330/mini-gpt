import unittest
import torch
from model import FeedForward, SwiGLUFeedForward, GPT
from activation_experiments import initialize, fingerprint_parameters

class FFNComparison(unittest.TestCase):
    def test_swiglu_contract_and_gradients(self):
        torch.manual_seed(42)
        for bias,total in ((False,1_179_648),(True,1_182_080)):
            m=SwiGLUFeedForward(384,1536,bias=bias)
            self.assertEqual(sum(p.numel() for p in m.parameters()),total)
            for shape in ((1,1,384),(2,7,384),(4,128,384)):
                x=torch.randn(*shape,requires_grad=True); original=x.detach().clone()
                y=m(x)
                self.assertEqual(y.shape,x.shape)
                torch.testing.assert_close(x.detach(),original)
                y.square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
            for name,p in m.named_parameters():
                self.assertIsNotNone(p.grad,name)
                self.assertTrue(torch.isfinite(p.grad).all(),name)
                self.assertTrue((p.grad!=0).any(),name)
    def test_correct_gate_branch(self):
        m=SwiGLUFeedForward(1,3)
        with torch.no_grad():
            m.gate_proj.weight.copy_(torch.tensor([[1.],[-1.]]))
            m.up_proj.weight.copy_(torch.tensor([[2.],[3.]]))
            m.down_proj.weight.copy_(torch.tensor([[1.,2.]]))
        x=torch.tensor([[[1.]]])
        expected=2*torch.sigmoid(torch.tensor(1.))-6*torch.sigmoid(torch.tensor(-1.))
        torch.testing.assert_close(m(x).squeeze(),expected)
    def test_plain_activation_math(self):
        for a,f in [('relu',torch.relu),('gelu',torch.nn.functional.gelu),('silu',torch.nn.functional.silu)]:
            m=FeedForward(8,12,a,bias=False); x=torch.randn(2,3,8)
            torch.testing.assert_close(m(x),m.fc2(f(m.fc1(x))))
    def test_defaults_and_compatible_weights(self):
        self.assertIsNone(SwiGLUFeedForward(8,12).gate_proj.bias)
        for a in ('relu','gelu','silu'):
            self.assertIsNotNone(FeedForward(8,12,a).fc1.bias)
        old=GPT(17,8,2,12,2)
        restored=GPT(17,8,2,12,2)
        restored.load_state_dict(old.state_dict(),strict=True)
    def test_architecture_selection_and_common_initialization(self):
        models=[]
        for a in ('relu','gelu','silu','swiglu'):
            m=GPT(17,8,2,12,2,activation=a,ffn_bias=False);initialize(m,42)
            self.assertEqual(sum(p.numel() for p in m.blocks[0].ffn.parameters()),192)
            self.assertEqual(m(torch.randint(0,17,(2,4))).shape,(2,4,17))
            self.assertEqual(isinstance(m.blocks[0].ffn,SwiGLUFeedForward),a=='swiglu')
            models.append(m)
        self.assertEqual(len({fingerprint_parameters(m,True) for m in models}),1)
        self.assertEqual(len({fingerprint_parameters(m) for m in models[:3]}),1)
if __name__=='__main__': unittest.main()
