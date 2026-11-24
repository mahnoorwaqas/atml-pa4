import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/"task1"))
from summarize import bootstrap_ppl,bootstrap_hs

class PairedBootstrapTests(unittest.TestCase):
    def test_window_pooled_nll_and_95pct_interval(self):
        baseline=[{"id":i,"tokens":10,"nll_sum":10.0} for i in range(20)]
        candidate=[{"id":i,"tokens":10,"nll_sum":11.0} for i in range(20)]
        result=bootstrap_ppl(candidate,baseline,rep=200,seed=6304)
        self.assertGreater(result['delta_ppl'],0)
        self.assertEqual(result['tokens'],200)
        self.assertEqual(result['ci95'][0],result['ci95'][1])
    def test_paired_hellaswag_char_norm(self):
        a=[{'order':i,'acc':1,'acc_norm':i%2} for i in range(20)]
        b=[{'order':i,'acc':0,'acc_norm':0} for i in range(20)]
        result=bootstrap_hs(a,b,'acc',rep=200)
        self.assertEqual(result['delta_accuracy'],1)
        self.assertAlmostEqual(bootstrap_hs(a,b,'acc_norm',200)['delta_accuracy'],.5)
    def test_reject_mismatched_paired_windows(self):
        a=[{'id':0,'tokens':10,'nll_sum':3.}]
        b=[{'id':1,'tokens':10,'nll_sum':3.}]
        with self.assertRaises(ValueError):
            bootstrap_ppl(a,b,rep=2)

if __name__=='__main__':unittest.main()
