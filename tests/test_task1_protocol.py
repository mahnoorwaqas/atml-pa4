import unittest
import torch
from task1 import protocol as p

class Task1ProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(6304)

    def test_affine_quantizer_shapes_and_exact_bytes(self):
        w=torch.randn(3,137).half()
        for bits in (2,3,4,8):
            z,nb=p.quantize(w,bits)
            self.assertEqual(z.shape,w.shape)
            self.assertEqual(z.dtype,w.dtype)
            self.assertTrue(torch.isfinite(z).all())
            self.assertEqual(nb,3*(16*bits+4+((9*bits+7)//8)+4))
            self.assertEqual(nb,p.payload_bytes(3,137,bits))

    def test_affine_calibration_independent_of_padding(self):
        w=torch.tensor([[1.,2.,3.,4.,5.]]).half()
        q,nb=p.quantize(w,8,group=4)
        self.assertTrue(torch.isfinite(q).all())
        self.assertEqual(nb,1*(4+4)+1*(1+4))

    def test_wanda_exact_zero_count_and_24(self):
        w=torch.arange(1,33,dtype=torch.float16).reshape(4,8)
        score=w.float()
        for n in (0,1,5,13,30,32):
            mask=p.ranked_mask(w,score,n)
            self.assertEqual(int((~mask).sum()),n)
        mask=p.ranked_mask(w,score,16,structured=True)
        self.assertEqual(int((~mask).sum()),16)
        self.assertTrue((mask.reshape(4,2,4).sum(-1)==2).all())

    def test_projection_bounded_exact(self):
        spec={'one':(800,0.1),'two':(400,3.0),'three':(1200,.4)}
        for target in (1200,1800):
            a=p.bounded_allocation(spec,target)
            self.assertEqual(sum(a.values()),target)
            self.assertTrue(all(.1<=a[k]/spec[k][0]<=.85 for k in a))
        with self.assertRaises(ValueError):
            p.bounded_allocation(spec,100)

    def test_mixed_bit_exact_budget(self):
        opts={'a':{2:(10.,20),3:(5.,30),4:(3.,40),8:(2.,80)},
              'b':{2:(20.,21),3:(7.,31),4:(4.,41),8:(3.,81)}}
        bits,used=p.mixed_allocation(opts,70)
        self.assertLessEqual(used,70)
        self.assertEqual(used,sum(opts[k][bits[k]][1] for k in bits))
        self.assertTrue(all(b in p.BITS for b in bits.values()))

    def test_awq_compensation_and_budget(self):
        w=torch.randn(4,17).half()
        features=torch.rand(17)+.01
        for b in (3,4,8):
            err,q,nbytes,alpha,preserve=p.awq(w,features,b)
            self.assertTrue(torch.isfinite(q).all())
            self.assertGreaterEqual(err,0)
            self.assertLess(preserve,1e-3)
            self.assertEqual(nbytes,p.payload_bytes(4,17,b)+2*17)
            self.assertIn(alpha,(0.,.25,.5,.75,1.))

if __name__=='__main__':unittest.main()
