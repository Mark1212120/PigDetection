"""Build all detection figures from saved results without training or inference."""
from pathlib import Path
import sys,argparse
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.visual_report import build_report
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--index',type=Path,default=ROOT/'experiments/final_results.json')
 p.add_argument('--output',type=Path,default=ROOT/'experiments/work_dirs/report')
 a=p.parse_args();print(build_report(a.index,a.output,ROOT))
if __name__=='__main__':main()
