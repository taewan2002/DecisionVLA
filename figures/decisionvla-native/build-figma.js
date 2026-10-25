const page = await figma.getNodeByIdAsync('1836:15');
await figma.setCurrentPageAsync(page);
const parent = await figma.getNodeByIdAsync('13099:47340');
const old = await figma.getNodeByIdAsync('13366:92873');
const name = 'DecisionVLA — architecture proposal';
if (parent.children.some(n => n.name === name)) throw new Error('Figure exists: inspect before retry');
await Promise.all(['Regular','Medium','Semi Bold','Bold'].map(style => figma.loadFontAsync({family:'Inter',style})));
const created = [];
const rgb = hex => ({r:parseInt(hex.slice(0,2),16)/255,g:parseInt(hex.slice(2,4),16)/255,b:parseInt(hex.slice(4,6),16)/255});
const paint = hex => [{type:'SOLID',color:rgb(hex)}];
const C={ink:'132B38',muted:'59717D',line:'B4C4CC',teal:'1B7C7A',blue:'3D6B9C',purple:'7759A5'};
function text(parent,label,size=18,style='Medium',color=C.ink) {const n=figma.createText();created.push(n.id);n.fontName={family:'Inter',style};n.fontSize=size;n.fills=paint(color);n.characters=label;parent.appendChild(n);return n;}
function box(parent,name,w,h,fill='FFFFFF',stroke=null){const n=figma.createFrame();created.push(n.id);n.name=name;n.resize(w,h);n.fills=paint(fill);n.cornerRadius=16;if(stroke){n.strokes=paint(stroke);n.strokeWeight=1;}parent.appendChild(n);return n;}
function stack(parent,name,w,gap=12){const n=figma.createAutoLayout('VERTICAL');created.push(n.id);n.name=name;n.fills=[];n.itemSpacing=gap;n.resize(w,100);n.primaryAxisSizingMode='AUTO';n.counterAxisSizingMode='FIXED';parent.appendChild(n);return n;}
function row(parent,name,gap=8){const n=figma.createAutoLayout('HORIZONTAL');created.push(n.id);n.name=name;n.fills=[];n.itemSpacing=gap;n.counterAxisAlignItems='CENTER';parent.appendChild(n);return n;}
function rect(parent,w,h,color,r=4){const n=figma.createRectangle();created.push(n.id);n.resize(w,h);n.fills=paint(color);n.cornerRadius=r;parent.appendChild(n);return n;}
function arrow(points,color=C.line){const n=figma.createVector();created.push(n.id);f.appendChild(n);n.vectorPaths=[{windingRule:'NONE',data:points}];n.fills=[];n.strokes=paint(color);n.strokeWeight=2;n.strokeCap='ROUND';return n;}
const f=box(parent,name,1320,550);f.x=3200;f.y=100;
const title=text(f,'DecisionVLA',32,'Bold');title.x=40;title.y=28;
const subtitle=text(f,'Parallel actions · visible uncertainty',17,'Regular',C.muted);subtitle.x=40;subtitle.y=72;
const status=box(f,'Proposal badge',224,34,'FFF4DF');status.x=1056;status.y=36;const st=text(status,'PROPOSED ARCHITECTURE',12,'Semi Bold','986312');st.x=18;st.y=10;
const input=stack(f,'Observation',188,12);input.x=40;input.y=145;
text(input,'Observation',17,'Semi Bold');
const img=old.findAll(n=>'fills' in n && Array.isArray(n.fills) && n.fills.some(p=>p.type==='IMAGE'))[0];
if(img){const copy=img.clone();input.appendChild(copy);copy.resize(170,140);copy.cornerRadius=10;created.push(copy.id);if('findAll' in copy) created.push(...copy.findAll(()=>true).map(n=>n.id));}
text(input,'“Pick up the bowl.”',15,'Medium');text(input,'RGB + instruction',13,'Regular',C.muted);
const backbone=box(f,'Shared native backbone',704,352,'F6F9FB','DCE6EB');backbone.x=270;backbone.y=122;
const label=text(backbone,'DiffusionGemma · shared Transformer + LoRA',17,'Semi Bold');label.x=24;label.y=20;
const prefix=box(backbone,'Observation prefix',202,170,'E8F1F8');prefix.x=24;prefix.y=76;
const ps=stack(prefix,'Encoder labels',162,14);ps.x=20;ps.y=23;text(ps,'Vision + prefix',20,'Semi Bold',C.blue);text(ps,'Encode once',14,'Regular',C.muted);const kv=row(ps,'Cached tokens',5);for(let i=0;i<7;i++)rect(kv,16,23,i%2?'AFC9E2':'C9DCEC',3);text(ps,'KV cache',14,'Semi Bold',C.blue);
const native=box(backbone,'Native loop',366,220,'EDF7F4','CAE3DB');native.x=304;native.y=76;
const ns=stack(native,'Denoiser and sampler',326,16);ns.x=20;ns.y=20;
text(ns,'Native denoiser',22,'Semi Bold',C.teal);text(ns,'Full vocabulary · self-conditioning',13,'Regular',C.muted);
const noise=row(ns,'Unrestricted canvas',6);for(let i=0;i<13;i++)rect(noise,16,20,['C2DDD5','B4D3CB','E2D7EE','C3D4E7'][i%4],3);
const sampler=box(ns,'Entropy bound',326,44,'D9EDE5');const sl=text(sampler,'Entropy-bound sampler',17,'Semi Bold',C.teal);sl.x=17;sl.y=12;
const looplabel=text(backbone,'Sample → accept → re-noise',13,'Medium',C.teal);looplabel.x=326;looplabel.y=315;
const output=stack(f,'Final readout',260,16);output.x=1010;output.y=145;
text(output,'Final action readout',19,'Semi Bold',C.purple);
const grid=stack(output,'Eight timesteps',250,4);for(let t=0;t<8;t++){const rr=row(grid,'Action '+t,4);for(let d=0;d<7;d++)rect(rr,21,11,['C4B5DD','D8CBE8','B7A4D3'][(t+d)%3],2);}
text(output,'8 × 7 actions',19,'Semi Bold');
const conf=row(output,'Probability bars',7);for(let i=0;i<7;i++)rect(conf,15,9+[18,7,28,13,22,5,16][i],'91B6B1',2);text(output,'Probabilities + uncertainty',14,'Medium',C.teal);
arrow('M 230 258 L 282 258 M 276 253 L 282 258 L 276 263');
arrow('M 498 258 L 570 258 M 564 253 L 570 258 L 564 263');
arrow('M 946 232 L 998 232 M 992 227 L 998 232 L 992 237',C.purple);
arrow('M 923 364 L 955 364 L 955 186 L 928 186 M 934 181 L 928 186 L 934 191',C.teal);
const final=text(f,'final pass',11,'Medium',C.purple);final.x=946;final.y=211;
const footer=text(f,'256-token canvas  ·  1 / 2 / 4 native passes to validate  ·  confidence ≠ success probability',14,'Regular',C.muted);footer.x=40;footer.y=505;
return {createdNodeIds:created,mutatedNodeIds:[parent.id],frame:{id:f.id,name:f.name,width:f.width,height:f.height},oldFigurePreserved:old.id,texts:f.findAllWithCriteria({types:['TEXT']}).map(n=>n.characters)};
