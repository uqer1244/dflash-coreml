"""Export selected JSON evidence without machine-specific absolute paths."""
import argparse
import json
import re
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[2])
args=parser.parse_args()
root=args.workspace.resolve()
out=Path(__file__).resolve().parents[1]/'docs/evidence'
out.mkdir(exist_ok=True)
files={
 'verification-summary.json':'ane-dflash/phase4_2/v2/summary.json',
 'verification-confirmation.json':'ane-dflash/phase4_2/v2/confirmation/report.json',
 'long-output-gate.json':'ane-dflash/phase4_2/v2/long128-results.json',
 'continuation-gate.json':'ane-dflash/phase4_2/v2/oracle-replay.json',
 'width-gate.json':'ane-dflash/phase4_2/v2/budget-gate.json',
}
def scrub(value):
 if isinstance(value,dict):return {scrub(k):scrub(v) for k,v in value.items()}
 if isinstance(value,list):return [scrub(v) for v in value]
 if isinstance(value,str):
  value=value.replace(str(root),'<research-workspace>')
  return re.sub(r'/Users/[^/\s]+','<user-home>',value)
 return value
for name,path in files.items():
 data=scrub(json.loads((root/path).read_text()))
 (out/name).write_text(json.dumps(data,indent=2)+'\n')
print(f'Exported {len(files)} evidence files')
