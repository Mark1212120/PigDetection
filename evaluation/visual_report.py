"""Publication figures from saved predictions and histories; never loads a detector."""
from pathlib import Path
import csv, json, hashlib, io
from contextlib import redirect_stdout
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

METHODS=['yolov8s','yolox_s','fasterrcnn_r50_fpn','retinanet_r50_fpn','fcos_r50_fpn','rtdetr_l']
NAMES=dict(zip(METHODS,['YOLOv8s','YOLOX-S','Faster R-CNN','RetinaNet','FCOS','RT-DETR-L']))
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def write_csv(path,rows):
 with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load_history(training,method):
 p=training/('engine/results.csv' if method in ['yolov8s','rtdetr_l'] else 'history.csv')
 with p.open(encoding='utf-8-sig',newline='') as f: rows=[{k.strip():float(v) for k,v in r.items()} for r in csv.DictReader(f)]
 if method in ['yolov8s','rtdetr_l']:
  valpath=training/'unified_validation.jsonl'
  vals=[readline for line in valpath.read_text(encoding='utf-8-sig').splitlines() if line.strip() for readline in [json.loads(line)]]
  ap={r['epoch']:r['metrics']['AP'] for r in vals}
  for r in rows:r['val_AP']=ap[int(r['epoch'])]
  losskeys=[k for k in rows[0] if k.startswith('train/')]
  for r in rows:r['train_loss']=sum(r[k] for k in losskeys)
 else:valpath=p
 return rows,p,valpath

def build_report(index_path,output,root=None):
 root=Path(root or Path(__file__).resolve().parents[1]);index_path=Path(index_path);out=Path(output)
 index=read(index_path);out.mkdir(parents=True,exist_ok=True)
 for f in [Path('C:/Windows/Fonts/times.ttf'),Path('C:/Windows/Fonts/timesbd.ttf')]:
  if f.is_file():font_manager.fontManager.addfont(str(f))
 family='Times New Roman' if any(f.name=='Times New Roman' for f in font_manager.fontManager.ttflist) else 'DejaVu Serif'
 plt.rcParams.update({'font.family':family,'font.size':10,'axes.titlesize':11,'axes.labelsize':10,
   'xtick.labelsize':8,'ytick.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,
   'pdf.fonttype':42,'axes.unicode_minus':False})
 sources={str(index_path):sha(index_path)};summary=[];validation=[];histories={};records={};figures=[]
 for m in METHODS:
  rec=index['models'][m];tr=root/rec['training_dir'];te=root/rec['test_dir']
  rm=read(tr/'run_manifest.json');metric=read(te/'metrics.json')
  if rm['status']!='completed' or rm['smoke_only'] or metric['smoke_only']:raise ValueError('Incomplete or smoke result: '+m)
  if rm['checkpoint_sha256']!=metric['weights_sha256']:raise ValueError('Checkpoint mismatch: '+m)
  history,hpath,vpath=load_history(tr,m)
  if len(history)!=rm['config']['epochs']:raise ValueError('Incomplete epoch history: '+m)
  best=max(history,key=lambda r:r['val_AP']);last=history[-1]
  metrics=metric['metrics'];timing=metric['timing']
  summary.append(dict(method=m,name=NAMES[m],**{k:metrics[k] for k in ['AP','AP50','AP75','AR100','AP_small','AP_medium','AP_large']},
    mean_ms=timing['mean_ms'],epochs=len(history),batch=rm['config']['batch'],effective_batch=rm['config']['effective_batch'],
    best_epoch=int(best['epoch']),best_val_AP=best['val_AP'],last_val_AP=last['val_AP'],
    AP_drop_pp=100*(best['val_AP']-last['val_AP'])))
  histories[m]=history;records[m]=metric
  validation.extend(dict(method=m,epoch=int(r['epoch']),val_AP=r['val_AP'],train_loss=r['train_loss']) for r in history)
  for p in [tr/'run_manifest.json',te/'metrics.json',te/'predictions.json',hpath,vpath]:sources[str(p)]=sha(p)
 write_csv(out/'comparison.csv',summary);write_csv(out/'training_curves.csv',validation)
 def save(fig,name):
  for fmt in ['png','pdf','svg']:fig.savefig(out/f'{name}.{fmt}',dpi=300,bbox_inches='tight',facecolor='white')
  plt.close(fig);figures.append(name)
 def grid(ax):ax.grid(axis='y',color='.88',linewidth=.5);ax.set_axisbelow(True)
 fig,ax=plt.subplots(figsize=(8,2.8),layout='constrained');x=np.arange(6);w=.19
 for i,(k,fc,hatch) in enumerate([('AP','white','///'),('AP50','.75',''),('AP75','.48',''),('AR100','.2','')]):
  bars=ax.bar(x+(i-1.5)*w,[100*r[k] for r in summary],w,label=k,facecolor=fc,edgecolor='black',lw=.6,hatch=hatch)
  for b in bars:ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,f'{b.get_height():.1f}',ha='center',fontsize=7)
 ax.set_xticks(x,[NAMES[m] for m in METHODS]);ax.set_ylabel('Score (%)');ax.set_ylim(0,111)
 ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,1.15),frameon=False);grid(ax);save(fig,'01_test_metrics')
 styles=['-','--','-.',':',(0,(5,1,1,1)),(0,(3,1,1,1,1,1))]
 fig,ax=plt.subplots(figsize=(8,2.7),layout='constrained')
 for m,style in zip(METHODS,styles):
  h=histories[m];ax.plot([r['epoch'] for r in h],[100*r['val_AP'] for r in h],ls=style,lw=1.2,label=NAMES[m])
  best=max(h,key=lambda r:r['val_AP']);ax.scatter(best['epoch'],100*best['val_AP'],s=15)
 ax.set_xlabel('Epoch');ax.set_ylabel('Validation COCO AP (%)');ax.legend(ncol=3,fontsize=8,frameon=False)
 grid(ax);save(fig,'02_validation_ap')
 fig,axes=plt.subplots(2,3,figsize=(8,4.2),layout='constrained')
 for ax,m in zip(axes.flat,METHODS):
  h=histories[m];ax.plot([r['epoch'] for r in h],[r['train_loss'] for r in h],color='black',lw=1)
  ax.set_title(NAMES[m]);ax.set_xlabel('Epoch');ax.set_ylabel('Native loss');grid(ax)
 save(fig,'03_training_losses')
 # Re-evaluate stored COCO predictions on CPU to recover PR curves; no inference.
 from pycocotools.coco import COCO
 from pycocotools.cocoeval import COCOeval
 dataset=root/index['dataset'];gtpath=dataset/'annotations/test.json';sources[str(gtpath)]=sha(gtpath)
 gtdata=read(gtpath);areas=[a['area'] for a in gtdata['annotations']]
 sizes=dict(small=sum(a<32**2 for a in areas),medium=sum(32**2<=a<96**2 for a in areas),large=sum(a>=96**2 for a in areas))
 pr=[]
 fig,axes=plt.subplots(1,2,figsize=(8,2.8),layout='constrained')
 for m,style in zip(METHODS,styles):
  pred=root/index['models'][m]['test_dir']/'predictions.json'
  with redirect_stdout(io.StringIO()):
   gt=COCO(str(gtpath));dt=gt.loadRes(str(pred));ev=COCOeval(gt,dt,'bbox')
   ev.params.catIds=[1];ev.params.maxDets=[1,10,100]
   ev.evaluate();ev.accumulate();ev.summarize()
  metric_keys=['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large']
  for stat,key in zip(ev.stats,metric_keys):
   if abs(float(stat)-records[m]['metrics'][key])>1e-9:raise ValueError('Saved COCO metric not reproduced: '+m+' '+key)
  for ax,threshold in zip(axes,[.5,.75]):
   ti=int(np.argmin(abs(ev.params.iouThrs-threshold)));v=ev.eval['precision'][ti,:,0,0,2]
   mask=v>=0;ax.plot(ev.params.recThrs[mask],v[mask],ls=style,label=NAMES[m],lw=1.1)
   pr.extend(dict(method=m,iou=threshold,recall=float(x),precision=float(y)) for x,y in zip(ev.params.recThrs[mask],v[mask]))
 for ax,t in zip(axes,[.5,.75]):
  ax.set_title(f'IoU = {t:.2f}');ax.set_xlabel('Recall');ax.set_ylabel('Interpolated precision')
  ax.set_xlim(0,1);ax.set_ylim(0,1.04);grid(ax)
 axes[0].legend(fontsize=7,frameon=False,loc='lower left')
 write_csv(out/'precision_recall.csv',pr);save(fig,'04_precision_recall')
 fig,ax=plt.subplots(figsize=(7,2.8),layout='constrained')
 for i,r in enumerate(summary):
  ax.scatter(r['mean_ms'],100*r['AP'],marker=['o','s','^','D','v','P'][i],s=38,color='black')
  offset=(-8,9) if r['method']=='fcos_r50_fpn' else (6,-12) if r['method']=='fasterrcnn_r50_fpn' else (4,4 if i%2==0 else -11)
  ax.annotate(NAMES[r['method']],(r['mean_ms'],100*r['AP']),xytext=offset,textcoords='offset points',fontsize=8,ha='right' if r['method']=='fcos_r50_fpn' else 'left')
 ax.set_xlabel('Batch 1 adapter latency (ms / image, excludes disk decode)');ax.set_ylabel('Test COCO AP (%)');ax.set_xlim(0,24)
 ax.set_ylim(65,78);grid(ax);save(fig,'05_latency_accuracy')
 fig,ax=plt.subplots(figsize=(8,2.5),layout='constrained');x=np.arange(6)
 ax.bar(x-.18,[100*r['AP_medium'] for r in summary],.36,color='white',edgecolor='black',hatch='///',label='Medium')
 ax.bar(x+.18,[100*r['AP_large'] for r in summary],.36,color='.5',edgecolor='black',label='Large')
 ax.set_xticks(x,[NAMES[m] for m in METHODS]);ax.set_ylabel('COCO AP (%)');ax.legend(frameon=False);grid(ax);save(fig,'06_scale_metrics')
 text=['# 猪只目标检测实验结果','',f"批次：{index['batch_id']}；设备：{index.get('hardware','见 metrics.json')}。",'',
  '以下分数由同一批既有预测文件核对。AP/AP50/AP75/AR100 为百分数；耗时为 batch=1 的预处理、模型与后处理均值，不含磁盘解码，不是论文端到端 FPS。','',
  '| 方法 | AP | AP50 | AP75 | AR100 | ms/图 | 最佳验证轮次 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
 for r in summary:text.append(f"| {r['name']} | {100*r['AP']:.2f} | {100*r['AP50']:.2f} | {100*r['AP75']:.2f} | {100*r['AR100']:.2f} | {r['mean_ms']:.2f} | {r['best_epoch']} |")
 size_note=f"测试集 {len(gtdata['images'])} 张、{len(areas):,} 个框，其中小目标 {sizes['small']} 个，中目标 {sizes['medium']} 个，大目标 {sizes['large']:,} 个。AP_small=-1 表示无定义，不能写成 0。"
 text+=['',size_note,
  '当前是同场景编号分块的探索性验证，不能证明新场景泛化或无语义近重复。单次 seed=42 没有统计显著性证据。',
  '训练损失来自各自原生模型，数值尺度不可直接横向排序。验证曲线均采用统一 pycocotools COCO AP。','']
 for name in figures:text.append(f'![{name}]({name}.png)\n')
 (out/'结果汇总.md').write_text('\n'.join(text),encoding='utf-8')
 import html
 table='<table><tr>'+''.join(f'<th>{v}</th>' for v in ['方法','AP','AP50','AP75','AR100','ms/图'])+'</tr>'
 for r in summary:table+='<tr>'+''.join(f'<td>{v}</td>' for v in [r['name'],*[f'{100*r[k]:.2f}' for k in ['AP','AP50','AP75','AR100']],f"{r['mean_ms']:.2f}"])+'</tr>'
 table+='</table>'
 body=''.join(f'<figure><img src="{n}.png" alt="{n}"><figcaption>{n}</figcaption></figure>' for n in figures)
 (out/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>猪只目标检测实验结果</title><style>body{max-width:1050px;margin:32px auto;padding:0 20px;font:16px/1.7 serif;color:#111}img{max-width:100%}table{border-collapse:collapse;width:100%}th,td{padding:8px;border-bottom:1px solid #ccc;text-align:center}figure{margin:35px 0}</style><h1>猪只目标检测实验结果</h1><p>'+html.escape(text[2])+'</p><p>'+html.escape(text[4])+'</p>'+table+'<p>'+html.escape(size_note)+'</p><p>同场景探索性验证；训练损失不可跨模型比较。</p>'+body+'</html>',encoding='utf-8')
 generated=[p for p in out.iterdir() if p.is_file() and p.name!='report_manifest.json']
 manifest=dict(status='completed',index=str(index_path),figures=figures,
  recomputed_saved_prediction_AP_matches=True,model_training_or_inference=False,source_sha256=sources,
  artifact_sha256={p.name:sha(p) for p in generated})
 (out/'report_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
 return out/'index.html'
