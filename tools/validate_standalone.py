"""Local research validation: public runtime + HF-style Core ML bundle, no app imports."""
import argparse,hashlib,json,sys
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--target',type=Path,required=True)
parser.add_argument('--bundle',type=Path,required=True)
parser.add_argument('--bridge',type=Path,required=True)
parser.add_argument('--workspace',type=Path,default=Path(__file__).resolve().parents[2])
args=parser.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dflash_coreml.generation import Generator
from dflash_coreml.runtime import VERSIONS
root=args.workspace.resolve()
fixtures={}
for name in ['short','medium','code','reasoning']:
 ref=json.loads((root/f'ane-dflash/phase4_2/v2/runs/long128-v1-2/{name}.json').read_text())
 fixtures[name]=ref
rows=[]
results={}
with Generator(args.target,args.bundle/'draft',args.bundle/'model-manifest.json',args.bridge) as generator:
 for name,ref in fixtures.items():
  out=generator.generate(ref['prompt'],max_tokens=128)
  same=out['token_ids']==ref['tokens'] and out['finish_reason']==ref['finish_reason']
  print(name,'PASS' if same else 'FAIL',out['generated_tokens'],flush=True)
  results[name]=out
  rows.append({'fixture':name,'tokens':out['generated_tokens'],'finish_reason':out['finish_reason'],
               'reference_final_ids_match':same,'output_tokens_per_second':out['output_tokens_per_second']})
  assert same,name
 # Same loaded instance receives a new prompt/reset and repeated rejection transactions.
 repeated=generator.generate(fixtures['short']['prompt'],max_tokens=32)
 assert repeated['token_ids']==fixtures['short']['tokens'][:32]
 # New mode yields a direct answer without the default thinking prefix; independent correctness scope.
 direct=generator.generate('What is the capital of South Korea? Answer in one sentence.',max_tokens=32,enable_thinking=False)
 assert direct['text'].strip() and 'Seoul' in direct['text']
report={'gate':'STANDALONE_GENERATION_PASS_ON_TESTED_FIXTURES','runtime_versions':VERSIONS,
        'scope':'new independent venv, public runtime, local HF-style converted bundle, four128-token fixtures vs research IDs/EOS, repeated reset and no-thinking smoke',
        'app_bundle_dependency':False,'rows':rows,'repeated_reset_32_ids_match':True,
        'no_thinking_smoke':direct['text'],'performance_scope':'one correctness run, not repeated speed proof'}
out=root/'release/docs/STANDALONE_VALIDATION.json';out.write_text(json.dumps(report,indent=2)+'\n')
local=root/'release/runs/validation-results.json';local.write_text(json.dumps(results,indent=2)+'\n')
print('STANDALONE validation PASS',flush=True)
