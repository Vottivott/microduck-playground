"""Aggregate independent seeded basketball evaluations, without hiding seed spread."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    groups=defaultdict(dict)
    for path in a.directory.rglob('*.json'):
        d=json.loads(path.read_text())
        if 'alive_time_weighted_metrics' not in d: continue
        label=path.name.split('_seed')[0]
        groups[label][d['seed']]=d
    result={}
    for name,by_seed in sorted(groups.items()):
        ds=list(by_seed.values()); n=sum(d['num_envs'] for d in ds)
        horizons={}
        for t in ds[0]['survival']:
            passed=sum(d['survival'][t]['survived'] for d in ds); rate=passed/n
            z=1.96; denom=1+z*z/n
            center=(rate+z*z/(2*n))/denom
            half=z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/denom
            horizons[t]={'survived':passed,'total':n,'fraction':rate,
                'binomial_95pct_interval':[center-half,center+half],
                'per_seed':{d['seed']:d['survival'][t]['fraction'] for d in ds}}
        failures=[f for d in ds for f in d['failures']]
        result[name]={'seeds':sorted(by_seed),'survival':horizons,
            'mean_across_seeds_metrics':{k:statistics.mean(d['alive_time_weighted_metrics'][k] for d in ds)
                for k in ds[0]['alive_time_weighted_metrics']},
            'failure_causes_can_overlap':{k:sum(f['causes'][k] for f in failures)
                for k in ['low_height','offset','tilt']}}
    text=json.dumps(result,indent=2)
    print(text)
    if a.out:a.out.write_text(text+'\n')


if __name__=='__main__':main()
