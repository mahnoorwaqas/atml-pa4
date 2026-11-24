import pathlib,sys,unittest
import torch

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/"task1"))
import run as t1

class ToyQwen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config=type('Cfg',(),{'num_hidden_layers':1})()
        for name in t1.ELIGIBLE:
            setattr(self,name,torch.nn.Linear(16,8,bias=False))

class RunnerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(6304)
        self.model=ToyQwen()
        self.modules=t1.modules(self.model)
        self.norms={k:torch.ones(16) for k in self.modules}
        self.sq={k:torch.ones(16) for k in self.modules}
        self.samples={k:torch.randn(12,16).half() for k in self.modules}

    def test_all_task1_conditions_cpu(self):
        for method in ('baseline','magnitude','wanda','nonuniform','structured','ptq','awq','mixed'):
            with self.subTest(method=method):
                m=ToyQwen()
                result=t1.apply(m,method,.5,4,self.norms,self.sq,self.samples)
                self.assertEqual(len(result['per_projection']),7)
                self.assertGreater(result['packed_total_bytes'],0)
                if method in ('magnitude','wanda','nonuniform','structured'):
                    self.assertEqual(result['eligible_zero_count'],
                                     round(result['eligible_weights']*.5))
                if method=='mixed':
                    self.assertLessEqual(result['eligible_q3_bytes'],result['eligible_q2_budget_bytes'])
                for value in result['per_projection'].values():
                    self.assertGreaterEqual(value['calibration_output_mse_sampled'],0)

if __name__=='__main__': unittest.main()
