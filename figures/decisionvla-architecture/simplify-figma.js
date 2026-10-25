// Editable native-vector composition. Preserves the public frame ID and a hidden backup.
const page = await figma.getNodeByIdAsync('1836:15');
await figma.setCurrentPageAsync(page);
await Promise.all(['Regular','Medium','Semi Bold','Bold'].map(style => figma.loadFontAsync({family:'Inter',style})));
const root=await figma.getNodeByIdAsync('13366:92873');
if(root.children.some(n=>n.name==='Compact / title')) return {alreadyBuilt:true,rootNodeId:root.id};
const parent=root.parent, created=[],mutated=[root.id];
function collect(n,out){out.push(n.id);if('children' in n)for(const c of n.children)collect(c,out);}
function fingerprint(n){
 function walk(v){const x={id:v.id,type:v.type,name:v.name};for(const k of ['x','y','width','height','visible','opacity','fills','strokes','strokeWeight','cornerRadius','vectorPaths'])if(k in v)x[k]=v[k];if(v.type==='TEXT')Object.assign(x,{characters:v.characters,fontName:v.fontName,fontSize:v.fontSize});if('children' in v)x.children=v.children.map(walk);return x;}
 const s=JSON.stringify(walk(n));let h=2166136261;for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619);}return (h>>>0).toString(16);
}
const protectedBefore=parent.children.filter(n=>n.id!==root.id).map(n=>({id:n.id,fingerprint:fingerprint(n)}));
const originalWords=root.query('TEXT').map(n=>n.characters).join(' ').split(/\s+/).length;
const photo=(await figma.getNodeByIdAsync('13366:92874')).clone();created.push(photo.id);
const backup=figma.createFrame();created.push(backup.id);parent.appendChild(backup);
backup.set({name:'Backup — DecisionVLA before compact architecture',x:root.x,y:root.y,width:root.width,height:root.height,visible:false,fills:root.fills});
for(const child of [...root.children]){backup.appendChild(child);mutated.push(child.id);}
root.set({name:'DecisionVLA — compact README architecture',width:1280,height:480,placeholder:true,fills:[{type:'SOLID',color:{r:1,g:1,b:1}}]});
const C={ink:'#15212C',blue:'#D5EAF4',blueDark:'#28678E',green:'#AFE258',greenSoft:'#E3F5C7',orange:'#F3C777',purple:'#E8DCF8',line:'#1D2B37',grey:'#E9EDF1'};
function rgb(h){return {r:parseInt(h.slice(1,3),16)/255,g:parseInt(h.slice(3,5),16)/255,b:parseInt(h.slice(5,7),16)/255};}
function paint(h){return [{type:'SOLID',color:rgb(h)}];}
function keep(n){created.push(n.id);return n;}
function label(parent,name,value,size=18,style='Semi Bold',color=C.ink){const t=keep(figma.createText());parent.appendChild(t);t.name=name;t.fontName={family:'Inter',style};t.fontSize=size;t.lineHeight={unit:'PERCENT',value:120};t.fills=paint(color);t.characters=value;t.textAutoResize='WIDTH_AND_HEIGHT';return t;}
function at(name,value,x,y,size=18,style='Semi Bold',color=C.ink){const t=label(root,name,value,size,style,color);t.x=x;t.y=y;return t;}
function rect(parent,name,x,y,w,h,color,stroke=null,r=4){const n=keep(figma.createRectangle());parent.appendChild(n);n.set({name,x,y,width:w,height:h,cornerRadius:r,fills:paint(color),strokes:stroke?paint(stroke):[],strokeWeight:1.4});return n;}
function block(name,value,x,y,w,h,color,size=18){const f=keep(figma.createAutoLayout('VERTICAL',{name,fills:paint(color),strokes:paint(C.line),strokeWeight:1.4,cornerRadius:7,primaryAxisAlignItems:'CENTER',counterAxisAlignItems:'CENTER'}));root.appendChild(f);f.resize(w,h);f.x=x;f.y=y;label(f,name+' / text',value,size);return f;}
function edge(name,points,color=C.line,dashed=false){const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),x=Math.min(...xs)-7,y=Math.min(...ys)-7,w=Math.max(...xs)-x+7,h=Math.max(...ys)-y+7;const end=points.at(-1),pre=points.at(-2),a=Math.atan2(end[1]-pre[1],end[0]-pre[0]),b1=[end[0]-8*Math.cos(a-.45),end[1]-8*Math.sin(a-.45)],b2=[end[0]-8*Math.cos(a+.45),end[1]-8*Math.sin(a+.45)];const d=points.map((p,i)=>(i?'L':'M')+(p[0]-x)+' '+(p[1]-y)).join(' ');const svg='<svg xmlns="http://www.w3.org/2000/svg" width="'+w+'" height="'+h+'" viewBox="0 0 '+w+' '+h+'"><path d="'+d+'" fill="none" stroke="'+color+'" stroke-width="1.8" stroke-linejoin="round" '+(dashed?'stroke-dasharray="5 4"':'')+'/><path d="M'+(b1[0]-x)+' '+(b1[1]-y)+' L'+(end[0]-x)+' '+(end[1]-y)+' L'+(b2[0]-x)+' '+(b2[1]-y)+'" fill="none" stroke="'+color+'" stroke-width="1.8"/></svg>';const n=figma.createNodeFromSvg(svg);collect(n,created);root.appendChild(n);n.name=name;n.x=x;n.y=y;return n;}
function pill(name,value,x,y,w,color){const f=block(name,value,x,y,w,23,color,12);f.strokes=[];return f;}
function tokenRow(parent,name,count,width,height,color){const row=keep(figma.createAutoLayout('HORIZONTAL',{name,itemSpacing:4,fills:[]}));parent.appendChild(row);for(let i=0;i<count;i++)rect(row,name+' '+i,0,0,width,height,Array.isArray(color)?color[i%color.length]:color,null,2);return row;}
at('Compact / title','DecisionVLA',232,15,30,'Bold');
rect(root,'Shared model boundary',224,62,804,357,'#FFFFFF',C.line,17);
at('Backbone label','DiffusionGemma 26B–A4B',248,80,18,'Bold');
at('Shared weights label','Shared Transformer weights',690,84,13,'Medium','#66717B');
root.appendChild(photo);photo.set({name:'Input / observation',x:40,y:105,width:134,height:134,cornerRadius:7});
at('Input / image label','RGB observation',34,249,16);
const instruction=block('Input / instruction','“Pick up the bowl.”',24,302,170,56,'#FFFFFF',15);
at('Input / instruction label','Task instruction',42,371,16);
rect(root,'Observation lane',250,132,240,257,'#FFFFFF','#9AABB8',10);
rect(root,'Observation lane label backing',260,111,218,27,'#FFFFFF',null,0);
const observationTitle=at('Observation lane title','Observation encoding',265,116,15,'Bold');
block('Frozen vision','Vision encoder',270,157,200,48,C.blue);
pill('Frozen badge','frozen',416,144,49,'#FFFFFF');
const visuals=tokenRow(root,'Visual tokens',8,17,17,C.blue);visuals.x=288;visuals.y=226;
block('Prefix pass','Prefix Transformer',270,285,200,58,C.blue,17);
pill('Prefix adapter','LoRA',410,272,50,C.purple);
block('Prefix cache','KV',512,292,66,44,C.grey,18);
at('Cache label','cache',524,342,13,'Medium');
rect(root,'Denoising lane',604,132,326,257,'#FFFFFF','#9AABB8',10);
rect(root,'Denoising label backing',615,111,193,27,'#FFFFFF',null,0);
at('Denoising lane title','Parallel denoising',621,116,15,'Bold');
const noise=tokenRow(root,'Noisy action tokens',8,21,20,['#EAE2CE','#D1DDE1','#E9ECCD','#E3D5E9']);noise.x=654;noise.y=157;
at('Noisy action label','Noisy actions',686,185,13,'Medium');
block('Action pass','Action Transformer',632,231,270,58,C.blue,19);
pill('Action adapter','LoRA',840,218,50,C.purple);
block('Readout','Action probabilities',632,327,270,42,C.greenSoft,18);
edge('RGB to vision',[[174,173],[270,173]]);
edge('Vision to tokens',[[370,205],[370,222]]);
edge('Tokens to prefix',[[370,246],[370,285]]);
edge('Instruction to prefix',[[194,330],[232,330],[232,314],[270,314]]);
edge('Prefix to KV',[[470,314],[512,314]]);
edge('KV to action',[[578,314],[592,314],[592,260],[632,260]]);
edge('Noise to action',[[765,205],[765,231]]);
edge('Action to readout',[[765,289],[765,327]]);
edge('Denoising repeat',[[902,348],[956,348],[956,167],[866,167]],'#8A939B',true);
at('Denoising iterations','×4',949,136,16,'Bold','#69737C');
edge('Probability branch',[[902,348],[996,348],[996,191],[1060,191]]);
edge('Confidence branch',[[996,348],[1060,348]]);
at('Action output title','8 × 7 actions',1076,112,19,'Bold');
const matrix=keep(figma.createAutoLayout('VERTICAL',{name:'Action chunk / schematic',itemSpacing:5,fills:[]}));root.appendChild(matrix);matrix.x=1080;matrix.y=151;
for(let r=0;r<7;r++)tokenRow(matrix,'Action row '+r,8,15,10,r===6?C.orange:C.green);
at('Confidence output title','Confidence',1085,285,19,'Bold');
const bars=keep(figma.createAutoLayout('HORIZONTAL',{name:'Confidence / schematic',itemSpacing:7,counterAxisAlignItems:'MAX',fills:[]}));root.appendChild(bars);bars.x=1090;bars.y=321;
for(const [i,height] of [20,37,28,43,32,39,24].entries())rect(bars,'Illustrative score '+i,0,0,13,height,C.orange,null,2);
at('Confidence qualifier','uncalibrated',1092,377,13,'Medium','#69737C');
const legend=keep(figma.createAutoLayout('HORIZONTAL',{name:'Compact legend',itemSpacing:10,fills:[]}));root.appendChild(legend);legend.x=250;legend.y=443;
rect(legend,'Frozen swatch',0,0,14,14,C.blue,null,2);label(legend,'Frozen legend','Frozen backbone',13,'Medium');
rect(legend,'LoRA swatch',0,0,14,14,C.purple,null,2);label(legend,'LoRA legend','Rank-32 LoRA',13,'Medium');
rect(legend,'Readout swatch',0,0,14,14,C.greenSoft,null,2);label(legend,'Readout legend','Trainable action rows',13,'Medium');
root.exportSettings=[{format:'SVG',svgOutlineText:true},{format:'PNG',constraint:{type:'SCALE',value:2}}];
root.placeholder=false;
const protectedAfter=protectedBefore.map(r=>({id:r.id,fingerprint:fingerprint(parent.children.find(n=>n.id===r.id))}));
await root.screenshot({scale:1});
return {rootNodeId:root.id,backupId:backup.id,createdNodeIds:created,mutatedNodeIds:mutated,protectedBefore,protectedAfter,protectedUnchanged:JSON.stringify(protectedBefore)===JSON.stringify(protectedAfter),wordsBefore:originalWords,wordsAfter:root.query('TEXT').map(n=>n.characters).join(' ').split(/\s+/).length,bounds:{width:root.width,height:root.height}};
