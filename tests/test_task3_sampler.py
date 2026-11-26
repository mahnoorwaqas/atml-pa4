"""100,000-sequence exact Markov χ² control and deliberately wrong residual."""
import collections,itertools,random,unittest
import numpy as np
from scipy.stats import chi2
from task3.sampler import positive_residual,exact_step
import torch

P=np.array([[.30,.45,.25],[.15,.50,.35],[.55,.20,.25],[.25,.35,.40]])
Q=np.array([[.65,.15,.20],[.55,.20,.25],[.15,.60,.25],[.65,.20,.15]])
N=100000
def draw(p,rng):
    u=rng.random();t=0.
    for i,x in enumerate(p):
        t+=float(x)
        if u<t:return i
    return len(p)-1
def generate(rng,gamma,wrong_residual=False):
    result=[]
    while len(result)<3:
        state=3 if not result else result[-1]
        ps=[];qs=[];proposed=[]
        for _ in range(min(gamma,3-len(result))):
            proposed.append(draw(Q[state],rng))
            qs.append(Q[state])
            ps.append(P[state])
            state=proposed[-1]
        ps.append(P[state])
        for i,tok in enumerate(proposed):
            if rng.random()<min(1.,float(ps[i][tok]/qs[i][tok])):
                continue
            residual=P[3 if i==0 and not result else (result[-1] if i==0 else proposed[i-1])] if wrong_residual else np.maximum(0,ps[i]-qs[i])
            if wrong_residual:
                residual=ps[i]
            else:residual=residual/residual.sum()
            result+=proposed[:i]+[draw(residual,rng)]
            break
        else:
            result+=proposed+[draw(ps[-1],rng)]
    return tuple(result[:3])
class ExactSamplerTests(unittest.TestCase):
    def test_residual(self):
        p=torch.tensor([.2,.7,.1]);q=torch.tensor([.6,.2,.2])
        r=positive_residual(p,q)
        self.assertTrue(torch.allclose(r,torch.tensor([0.,1.,0.]),atol=1e-6))
    def test_acceptance_and_bonus(self):
        p=[torch.tensor([.5,.5])]*3
        q=[torch.tensor([.5,.5])]*2
        out,accepted,examined,bonus=exact_step(p,q,[0,1],lambda:0.5,lambda probs:0)
        self.assertEqual(out,[0,1,0]);self.assertEqual(accepted,2)
        self.assertEqual(examined,2);self.assertTrue(bonus)
        p=[torch.tensor([.1,.9])]*3
        q=[torch.tensor([.9,.1])]*2
        out,accepted,examined,bonus=exact_step(p,q,[0,1],lambda:0.95,lambda probs:1)
        self.assertEqual(out,[1]);self.assertEqual(accepted,0)
        self.assertEqual(examined,1);self.assertFalse(bonus)
    def test_100000_markov_chisquare_and_negative_control(self):
        categories=list(itertools.product(range(3),repeat=3))
        expected={}
        for seq in categories:
            expected[seq]=N*P[3,seq[0]]*P[seq[0],seq[1]]*P[seq[1],seq[2]]
        self.assertGreaterEqual(min(expected.values()),10.)
        observed=collections.Counter(generate(random.Random(6304+i),2) for i in range(N))
        bad=collections.Counter(generate(random.Random(6404+i),2,True) for i in range(N))
        def test(counts):
            stat=sum((counts[key]-expected[key])**2/expected[key] for key in categories)
            return float(chi2.sf(stat,len(categories)-1)),stat
        correct_p,stat=test(observed)
        wrong_p,wrong_stat=test(bad)
        print(f'100k Markov joint: exact χ²={stat:.3f}, p={correct_p:.5g}; '
              f'wrong-residual χ²={wrong_stat:.3f}, p={wrong_p:.5g}')
        self.assertGreaterEqual(correct_p,.01)
        self.assertLess(wrong_p,.01)
if __name__=='__main__':unittest.main()
