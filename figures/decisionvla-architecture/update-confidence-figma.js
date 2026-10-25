const page = await figma.getNodeByIdAsync('1836:15');
await figma.setCurrentPageAsync(page);
const root = await figma.getNodeByIdAsync('13366:92873');
if(root.children.some(n=>n.name==='Confidence / probabilities')) return {alreadyUpdated:true,rootNodeId:root.id};
const fonts = root.query('TEXT').toArray().flatMap(t=>t.getStyledTextSegments(['fontName']).map(s=>s.fontName));
await Promise.all(fonts.map(font=>figma.loadFontAsync(font)));
const created=[],mutated=[];
const paint=hex=>[{type:'SOLID',color:{r:parseInt(hex.slice(1,3),16)/255,g:parseInt(hex.slice(3,5),16)/255,b:parseInt(hex.slice(5,7),16)/255}}];
const keep=n=>{created.push(n.id);return n;};
function txt(parent,name,value,size=18,style='Regular',width=450,color='#375D63'){
 const t=keep(figma.createText());parent.appendChild(t);t.set({name,fontName:{family:'Inter',style},fontSize:size,lineHeight:{unit:'PERCENT',value:130},fills:paint(color)});t.characters=value;t.resize(width,30);t.textAutoResize='HEIGHT';return t;
}
function card(name,x,w,title,lines){
 const f=keep(figma.createAutoLayout('VERTICAL',{name,itemSpacing:10,paddingLeft:22,paddingRight:22,paddingTop:20,paddingBottom:20,fills:paint('#EDF8F8'),strokes:paint('#8EB8BD'),strokeWeight:1.5,cornerRadius:14}));root.appendChild(f);f.resize(w,186);f.x=x;f.y=770;
 txt(f,name+' / title',title,23,'Semi Bold',w-44,'#245C63');
 for(let i=0;i<lines.length;i++)txt(f,name+' / detail '+i,lines[i],18,'Regular',w-44);
 return f;
}
function path(name,pts,arrow=true){
 const x=Math.min(...pts.map(p=>p[0])),y=Math.min(...pts.map(p=>p[1]));
 const v=keep(figma.createVector());root.appendChild(v);v.name=name;v.vectorPaths=[{windingRule:'NONE',data:pts.map((p,i)=>(i?'L ':'M ')+(p[0]-x)+' '+(p[1]-y)).join(' ')}];v.x=x;v.y=y;v.fills=[];v.strokes=paint('#3E7B82');v.strokeWeight=2;v.strokeJoin='ROUND';
 if(arrow){const e=pts.at(-1),p=pts.at(-2),a=Math.atan2(e[1]-p[1],e[0]-p[0]);path(name+' / arrowhead',[[e[0]-9*Math.cos(a-.48),e[1]-9*Math.sin(a-.48)],e,[e[0]-9*Math.cos(a+.48),e[1]-9*Math.sin(a+.48)]],false);}
}
root.resize(1760,1180);mutated.push(root.id);
const subtitle=await figma.getNodeByIdAsync('13367:92874');subtitle.characters='Direct robot actions with observable decision uncertainty';mutated.push(subtitle.id);
for(const id of ['13367:93001','13367:93005']){const n=await figma.getNodeByIdAsync(id);n.y=990;mutated.push(id);}
const note=await figma.getNodeByIdAsync('13367:93009');note.set({x:48,y:696,fontSize:17});note.resize(806,45);note.textAutoResize='HEIGHT';note.characters='Native 256-token canvas: 56 action slots + 200 fixed padding slots\nPadding slots are excluded from attention keys';mutated.push(note.id);
const branchLabel=txt(root,'Confidence / branch label','Final logits → softmax · no extra learned module',18,'Medium',710);branchLabel.x=925;branchLabel.y=703;
card('Confidence / probabilities',48,530,'Final-step probabilities',[
 '8 × 7 × 256 · forbidden gripper bins = 0',
 'Conditional on the sampled denoising path',
 'Read from the existing final forward pass'
]);
card('Confidence / scores',644,510,'Action-wise confidence',[
 'Continuous: distance-aware confidence',
 'Gripper: normalized peak probability',
 'Raw, uncalibrated scores'
]);
card('Confidence / evaluation',1220,492,'OFFLINE · LIBERO failure audit',[
 'First-query uncertainty → final outcome',
 'Within-task AUROC + bootstrap interval',
 'Executed actions remain unchanged'
]);
path('Readout → probability diagnostics',[[922,603],[900,603],[900,740],[313,740],[313,770]]);
path('Probabilities → confidence',[[578,863],[644,863]]);
path('Confidence → failure analysis',[[1154,863],[1220,863]]);
root.exportSettings=[{format:'PNG',constraint:{type:'SCALE',value:2}},{format:'SVG',svgOutlineText:true}];
await root.screenshot({scale:1});
return {rootNodeId:root.id,createdNodeIds:created,mutatedNodeIds:mutated,bounds:{width:root.width,height:root.height}};
