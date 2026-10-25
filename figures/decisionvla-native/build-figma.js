// In-place Figma authoring source; keeps a hidden backup of the previous figure.
await figma.setCurrentPageAsync(await figma.getNodeByIdAsync('1836:15'));
const parent=await figma.getNodeByIdAsync('13099:47340');
const f=await figma.getNodeByIdAsync('13407:92873');
const backupName='DecisionVLA — before figure polish 2026-10-25';
if(parent.children.some(n=>n.name===backupName))throw new Error('Polish already applied; inspect before retry.');
const fonts=new Map();
for(const n of f.findAllWithCriteria({types:['TEXT']}))for(const s of n.getStyledTextSegments(['fontName']))fonts.set(JSON.stringify(s.fontName),s.fontName);
for(const style of ['Regular','Medium','Semi Bold','Bold'])fonts.set(style,{family:'Inter',style});
await Promise.all([...fonts.values()].map(font=>figma.loadFontAsync(font)));
function snapshot(n){const a={id:n.id,name:n.name,type:n.type,x:n.x,y:n.y,width:n.width,height:n.height,visible:n.visible};if('fills'in n)a.fills=n.fills;if('strokes'in n)a.strokes=n.strokes;if(n.type==='TEXT'){a.characters=n.characters;a.fontSize=n.fontSize;a.fontName=n.fontName;}if('vectorPaths'in n)a.vectorPaths=n.vectorPaths;if('children'in n)a.children=n.children.map(snapshot);return a;}
const neighbors=parent.children.filter(n=>n.id!==f.id);
const before=JSON.stringify(neighbors.map(snapshot));
const imageNode=f.findAll(n=>'fills'in n&&Array.isArray(n.fills)&&n.fills.some(p=>p.type==='IMAGE'))[0];
if(!imageNode)throw new Error('Existing LIBERO observation is required.');
const imageFills=imageNode.fills;
const backup=f.clone();parent.appendChild(backup);backup.name=backupName;backup.visible=false;
const created=[backup.id,...backup.findAll(()=>true).map(n=>n.id)];
const removed=f.findAll(()=>true).map(n=>n.id);
for(const n of [...f.children])n.remove();
const rgb=h=>({r:parseInt(h.slice(0,2),16)/255,g:parseInt(h.slice(2,4),16)/255,b:parseInt(h.slice(4,6),16)/255});
const paint=h=>[{type:'SOLID',color:rgb(h)}];
const C={ink:'172A36',muted:'5A6D78',line:'6F8490',blue:'DCECF7',blueDark:'345C78',teal:'DAEFE8',tealDark:'246F62',violet:'75629E'};
f.name='DecisionVLA — architecture';f.resize(1360,424);f.cornerRadius=0;f.fills=paint('FFFFFF');
function add(n,p,name){created.push(n.id);p.appendChild(n);n.name=name;return n;}
function text(p,label,size=16,style='Medium',color=C.ink){const n=add(figma.createText(),p,label);n.fontName={family:'Inter',style};n.fontSize=size;n.characters=label;n.fills=paint(color);return n;}
function label(p,s,x,y,size=16,style='Medium',color=C.ink){const n=text(p,s,size,style,color);n.x=x;n.y=y;return n;}
function rect(p,name,x,y,w,h,fill,stroke=null,r=5){const n=add(figma.createRectangle(),p,name);n.resize(w,h);n.x=x;n.y=y;n.cornerRadius=r;n.fills=fill?paint(fill):[];if(stroke){n.strokes=paint(stroke);n.strokeWeight=1.2;}return n;}
function stack(p,name,x,y,gap=8,dir='VERTICAL'){const n=add(figma.createAutoLayout(dir),p,name);n.fills=[];n.itemSpacing=gap;n.primaryAxisSizingMode='AUTO';n.counterAxisSizingMode='AUTO';n.x=x;n.y=y;return n;}
function path(name,data,color=C.line,width=1.7){const n=add(figma.createVector(),f,name);n.vectorPaths=[{windingRule:'NONE',data}];n.fills=[];n.strokes=paint(color);n.strokeWeight=width;n.strokeCap='ROUND';n.strokeJoin='ROUND';return n;}
function centeredBlock(p,name,w,h,fill,stroke=null){const n=add(figma.createAutoLayout('VERTICAL'),p,name);n.resize(w,h);n.primaryAxisSizingMode='FIXED';n.counterAxisSizingMode='FIXED';n.primaryAxisAlignItems='CENTER';n.counterAxisAlignItems='CENTER';n.itemSpacing=8;n.cornerRadius=5;n.fills=paint(fill);if(stroke){n.strokes=paint(stroke);n.strokeWeight=1.2;}return n;}
label(f,'DecisionVLA',230,27,28,'Bold');
rect(f,'One shared backbone',206,84,898,304,'FFFFFF','536875',12);
label(f,'DiffusionGemma · shared Transformer + LoRA',230,101,17,'Semi Bold');
const observation=stack(f,'Observation and instruction',28,129,12);
const img=rect(observation,'LIBERO observation',0,0,142,142,null,'B7C5CC',6);img.fills=imageFills;
const instruction=text(observation,'“Pick up the bowl.”',14,'Medium');instruction.textAutoResize='HEIGHT';instruction.resize(142,36);
path('Observation to encoder','M 174 168 L 224 168 M 218 164 L 224 168 L 218 172');
path('Instruction to prefix','M 174 292 L 215 292 L 215 260 L 224 260 M 218 256 L 224 260 L 218 264');
const prefix=stack(f,'Vision and cached prefix',230,146,36);
const vision=centeredBlock(prefix,'Vision encoder',176,44,C.blue,'91B3CB');text(vision,'Vision encoder',16,'Semi Bold',C.blueDark);
const prefixLayers=stack(prefix,'Prefix Transformer layers',0,0,4);
for(let i=0;i<3;i++){const layer=centeredBlock(prefixLayers,'Prefix layer '+i,176,20,i===1?'D8E9F4':'ECF4F9','B1CBDC');if(i===1)text(layer,'Prefix encoding',14,'Semi Bold',C.blueDark);}
path('Vision to prefix','M 318 191 L 318 220 M 314 214 L 318 220 L 322 214');
const kv=stack(f,'Cached KV tokens',251,312,5,'HORIZONTAL');
for(let i=0;i<7;i++)rect(kv,'Cached token '+i,0,0,14,19,i%2?'ADCBDD':'C8DDEA',null,2);
label(f,'KV cache',284,339,13,'Medium',C.blueDark);
path('Prefix to KV cache','M 318 296 L 318 307 M 314 301 L 318 307 L 322 301');
path('Cached context to denoiser','M 385 321 L 460 321 L 460 260 L 509 260 M 503 256 L 509 260 L 503 264',C.blueDark);
label(f,'cached context',412,238,12,'Medium',C.blueDark);
const canvas=stack(f,'Noisy canvas',529,153,5,'HORIZONTAL');
for(let i=0;i<13;i++)rect(canvas,'Canvas position '+i,0,0,12,19,['C2DCD3','B8CFE1','DDD1EB','DAE6DE'][i%4],null,2);
label(f,'Noisy canvas',573,130,13,'Medium',C.muted);
path('Canvas to denoiser','M 631 174 L 631 197 M 627 191 L 631 197 L 635 191');
rect(f,'Transformer layer back',523,202,216,70,'ECF4F9','A6C5D9',5);
rect(f,'Transformer layer middle',519,208,224,70,'E5F0F7','9DBED4',5);
const denoiser=centeredBlock(f,'Native denoiser',232,70,C.blue,'80A8C4');denoiser.x=515;denoiser.y=216;
text(denoiser,'Native denoiser',19,'Semi Bold',C.blueDark);text(denoiser,'Full vocabulary',13,'Regular',C.muted);
path('Denoiser to sampler','M 750 251 L 817 251 M 811 247 L 817 251 L 811 255',C.tealDark);
const sampler=centeredBlock(f,'Entropy-bound sampler',244,78,C.teal,'92C4B5');sampler.x=824;sampler.y=212;
text(sampler,'Entropy-bound sampler',16,'Semi Bold',C.tealDark);
const accepted=stack(sampler,'Accepted and resampled tokens',0,0,5,'HORIZONTAL');
for(let i=0;i<10;i++)rect(accepted,'Sampled position '+i,0,0,14,15,[2,5,8].includes(i)?'E9F3EF':'8FBFB0',null,2);
path('Iterative refinement','M 946 294 L 946 344 Q 946 350 940 350 L 637 350 Q 631 350 631 344 L 631 292 M 627 298 L 631 292 L 635 298',C.tealDark);
label(f,'Self-conditioning + re-noise',701,361,13,'Medium',C.tealDark);
path('Final readout branch','M 1072 251 L 1136 251 L 1136 193 L 1154 193 M 1148 189 L 1154 193 L 1148 197 M 1136 251 L 1136 317 L 1154 317 M 1148 313 L 1154 317 L 1148 321',C.violet);
const actions=stack(f,'Parallel action readout',1163,128,12);
text(actions,'8 × 7 actions',18,'Semi Bold',C.ink);
const grid=stack(actions,'Eight timesteps and seven controls',0,0,4);
for(let t=0;t<8;t++){const row=stack(grid,'Timestep '+t,0,0,4,'HORIZONTAL');for(let d=0;d<7;d++)rect(row,'Action '+t+','+d,0,0,17,9,['BEB0D7','D9CFE8','AD9BCB'][(t+d)%3],null,2);}
const dist=stack(f,'Schematic action distribution',1163,283,5,'HORIZONTAL');dist.counterAxisAlignItems='MAX';
for(let i=0;i<9;i++)rect(dist,'Probability bin '+i,0,0,12,[5,9,19,34,48,40,25,13,6][i],'8BB8AB',null,2);
label(f,'Probabilities + confidence',1163,342,13,'Semi Bold',C.tealDark);
f.exportSettings=[{format:'PNG',suffix:'',constraint:{type:'SCALE',value:2}}];
const texts=f.findAllWithCriteria({types:['TEXT']});const bounds=f.absoluteBoundingBox;
const overflow=f.findAll(n=>{const b=n.absoluteBoundingBox;return b&&(b.x<bounds.x-1||b.y<bounds.y-1||b.x+b.width>bounds.x+bounds.width+1||b.y+b.height>bounds.y+bounds.height+1);}).map(n=>({id:n.id,name:n.name}));
const unchanged=JSON.stringify(neighbors.map(snapshot))===before;
if(!unchanged||overflow.length)throw new Error('Figure validation failed: '+JSON.stringify({unchanged,overflow}));
return {createdNodeIds:created,mutatedNodeIds:[f.id,parent.id],removedNodeIds:removed,backup:backup.id,frame:{id:f.id,width:f.width,height:f.height},neighborsUnchanged:unchanged,overflow,texts:texts.map(n=>n.characters),textNodeCount:texts.length,wordCount:texts.map(n=>n.characters).join(' ').split(/\s+/).length};
